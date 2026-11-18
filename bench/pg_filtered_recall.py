"""Measure REAL pgvector HNSW filtered-search recall for the three tenancy models.

    POSTGRES_DATABASE_URL=postgresql://... python bench/pg_filtered_recall.py --tenants 100 1000

Isolated experiment tables (dropped at the end, never touching production tables):
  shared     one table + one HNSW index, ``WHERE tenant_id = X ORDER BY emb <=> q``
  dedicated  one table + one HNSW index per tenant
  hybrid     tenants >= --big chunks dedicated, the rest in one shared table

Ground truth is the exact top-10 by cosine within the tenant (numpy). Recall is the
overlap of pgvector's answer with that truth. pgvector applies the tenant predicate
AFTER the HNSW scan returns ``hnsw.ef_search`` candidates, so selective filters lose
results. (Iterative scans, pgvector >= 0.8, are a mitigation; this reports the version
it ran on.) Additionally checks whether the PRODUCTION retrieval query's plan uses the
HNSW index at all.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text

from app.infrastructure.postgres.tenant_db import create_postgres_engine, tenant_transaction
from tenantrag.corpus.simulator import build_corpus
from tenantrag.eval.metrics import percentile
from tenantrag.tenancy.pg import _DENSE, PgTenantStore
from tenantrag.text import HashingEmbedder

DIM = 768


def _load(conn, table: str, ids: np.ndarray, vecs: np.ndarray, tenant: str | None):
    ins = text(f"INSERT INTO {table} (" + ("tenant_id, " if tenant is not None else "") + "chunk_id, embedding) VALUES ("
               + (":t, " if tenant is not None else "") + ":i, :e)").bindparams(bindparam("e", type_=Vector(DIM)))
    rows = [({"t": tenant} if tenant is not None else {}) | {"i": int(i), "e": v.tolist()} for i, v in zip(ids, vecs)]
    conn.execute(ins, rows)


def run(n_tenants: int, n_probe: int, seed: int, big: int) -> list[dict]:
    engine = create_postgres_engine(os.environ["POSTGRES_DATABASE_URL"], pool_size=4)
    emb = HashingEmbedder(DIM)
    corpus = build_corpus(n_tenants=n_tenants, max_docs=300, min_docs=6, alpha=0.9, queries_per_tenant=4, seed=seed)
    chunks = corpus.chunks
    vecs = emb.embed([c.text for c in chunks])
    by_t: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        by_t.setdefault(c.tenant_id, []).append(i)
    with engine.connect() as c:
        pgv = c.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'")).scalar_one()
    print(f"[{n_tenants} tenants] {len(chunks)} chunks, pgvector {pgv}", flush=True)

    probes = [g for g in corpus.golden if g.kind == "answerable"]
    random.Random(seed).shuffle(probes)
    probes = probes[:n_probe]
    qvecs = [emb.embed_query(g.query) for g in probes]
    truth = []
    for g, q in zip(probes, qvecs):
        idx = np.array(by_t[g.tenant_id])
        top = idx[np.argsort(-(vecs[idx] @ q))[:10]]
        truth.append({int(chunks[i].chunk_id) for i in top})

    with engine.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS bench_shared, bench_small_shared CASCADE"))
        for t in range(n_tenants):
            c.execute(text(f"DROP TABLE IF EXISTS bench_t_{t}"))

    def idx_sql(table):
        return f"CREATE INDEX ON {table} USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"

    built: dict[str, dict] = {}
    # shared
    t0 = time.perf_counter()
    with engine.begin() as c:
        c.execute(text(f"CREATE TABLE bench_shared (tenant_id text NOT NULL, chunk_id bigint NOT NULL, embedding vector({DIM}))"))
        for t, idxs in by_t.items():
            _load(c, "bench_shared", np.array([chunks[i].chunk_id for i in idxs]), vecs[idxs], t)
        c.execute(text(idx_sql("bench_shared")))
        sz = c.execute(text("SELECT pg_total_relation_size('bench_shared')")).scalar_one()
    built["shared"] = {"build_s": time.perf_counter() - t0, "bytes": sz, "relations": 1}
    # dedicated (per tenant) and hybrid's dedicated subset
    tid_num = {t: i for i, t in enumerate(by_t)}
    t0 = time.perf_counter()
    ded_bytes = 0
    with engine.begin() as c:
        for t, idxs in by_t.items():
            c.execute(text(f"CREATE TABLE bench_t_{tid_num[t]} (chunk_id bigint NOT NULL, embedding vector({DIM}))"))
            _load(c, f"bench_t_{tid_num[t]}", np.array([chunks[i].chunk_id for i in idxs]), vecs[idxs], None)
            c.execute(text(idx_sql(f"bench_t_{tid_num[t]}")))
        ded_bytes = c.execute(text("SELECT COALESCE(sum(pg_total_relation_size(oid)),0) FROM pg_class WHERE relname LIKE 'bench\\_t\\_%' AND relkind='r'")).scalar_one()
    built["dedicated"] = {"build_s": time.perf_counter() - t0, "bytes": int(ded_bytes), "relations": len(by_t)}
    big_t = {t for t, i in by_t.items() if len(i) >= big}
    small = [i for t, ii in by_t.items() if t not in big_t for i in ii]
    t0 = time.perf_counter()
    with engine.begin() as c:
        c.execute(text(f"CREATE TABLE bench_small_shared (tenant_id text NOT NULL, chunk_id bigint NOT NULL, embedding vector({DIM}))"))
        for t, idxs in by_t.items():
            if t not in big_t:
                _load(c, "bench_small_shared", np.array([chunks[i].chunk_id for i in idxs]), vecs[idxs], t)
        if small:
            c.execute(text(idx_sql("bench_small_shared")))
        small_bytes = c.execute(text("SELECT pg_total_relation_size('bench_small_shared')")).scalar_one()
        big_bytes = c.execute(text(
            "SELECT COALESCE(sum(pg_total_relation_size(oid)),0) FROM pg_class WHERE relname = ANY(:n) AND relkind='r'"),
            {"n": [f"bench_t_{tid_num[t]}" for t in big_t]}).scalar_one()
    built["hybrid"] = {"build_s": built["dedicated"]["build_s"] * len(big_t) / max(1, len(by_t)), "bytes": int(small_bytes + big_bytes),
                       "relations": len(big_t) + 1}
    c = None

    q_shared = text("SELECT chunk_id FROM bench_shared WHERE tenant_id = :t ORDER BY embedding <=> :q LIMIT 10").bindparams(bindparam("q", type_=Vector(DIM)))
    q_small = text("SELECT chunk_id FROM bench_small_shared WHERE tenant_id = :t ORDER BY embedding <=> :q LIMIT 10").bindparams(bindparam("q", type_=Vector(DIM)))

    def measure(mode: str, ef: int) -> dict:
        rec, lat, empty = [], [], 0
        with engine.connect() as conn:
            conn.execute(text("SET enable_seqscan = off"))  # make the planner prefer the ANN index
            conn.execute(text(f"SET hnsw.ef_search = {ef}"))
            for g, q, tr in zip(probes, qvecs, truth):
                t0 = time.perf_counter()
                if mode == "shared" or (mode == "hybrid" and g.tenant_id not in big_t):
                    stmt = q_shared if mode == "shared" else q_small
                    got = {r[0] for r in conn.execute(stmt, {"t": g.tenant_id, "q": q.tolist()})}
                else:
                    stmt = text(f"SELECT chunk_id FROM bench_t_{tid_num[g.tenant_id]} ORDER BY embedding <=> :q LIMIT 10").bindparams(bindparam("q", type_=Vector(DIM)))
                    got = {r[0] for r in conn.execute(stmt, {"q": q.tolist()})}
                lat.append((time.perf_counter() - t0) * 1e3)
                empty += not got
                rec.append(len(got & tr) / len(tr))
        b = built[mode]
        return {"tenants": n_tenants, "chunks": len(chunks), "mode": mode, "ef_search": ef,
                "recall_vs_exact@10": round(float(np.mean(rec)), 3), "empty_results": empty,
                "p95_ms": round(percentile(lat, 95), 2), "index_MB": round(b["bytes"] / 1e6, 1),
                "relations": b["relations"], "build_s": round(b["build_s"], 1), "pgvector": pgv}

    rows = []
    for mode, efs in (("dedicated", (40,)), ("hybrid", (40,)), ("shared", (40, 100, 400, 1000))):
        for ef in efs:
            rows.append(measure(mode, ef))
            print(rows[-1], flush=True)

    # does the PRODUCTION query shape (tenant + ACL join) use the HNSW index at all?
    store = PgTenantStore(engine, emb)
    sub = sorted(by_t)[: min(n_tenants, 40)]
    with engine.begin() as c:
        c.execute(text("DELETE FROM tenants WHERE tenant_id = ANY(:t)"), {"t": sub})
    store.ingest_corpus(corpus, tenants=sub)
    with engine.begin() as c:
        c.execute(text("ANALYZE document_chunks"))
    g = next(g for g in corpus.golden if g.tenant_id in sub)
    p = g.principal
    params = {"tid": p.tenant_id, "uid": str(p.user_id), "gids": [str(i) for i in p.group_ids], "roles": ["tenant_member"],
              "n": 10, "emb": emb.embed_query(g.query).tolist()}
    with tenant_transaction(engine, p.tenant_id) as conn:
        plan = "\n".join(r[0] for r in conn.execute(
            text("EXPLAIN " + str(_DENSE)).bindparams(bindparam("emb", type_=Vector(DIM))), params))
    rows.append({"tenants": n_tenants, "mode": "production_query_plan", "uses_hnsw": "hnsw" in plan.lower(),
                 "note": "planner drives from documents(tenant, ACL) then exact-sorts the tenant's chunks"
                         if "hnsw" not in plan.lower() else "HNSW index scan"})
    print(rows[-1], flush=True)

    with engine.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS bench_shared, bench_small_shared CASCADE"))
        for t in range(n_tenants):
            c.execute(text(f"DROP TABLE IF EXISTS bench_t_{t}"))
        c.execute(text("DELETE FROM tenants WHERE tenant_id = ANY(:t)"), {"t": sub})
    engine.dispose()
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenants", type=int, nargs="+", default=[100, 1000])
    ap.add_argument("--probes", type=int, default=200)
    ap.add_argument("--big", type=int, default=300, help="hybrid: chunks at which a tenant gets its own index")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rows = [r for n in a.tenants for r in run(n, a.probes, a.seed, a.big)]
    Path(a.out).mkdir(exist_ok=True)
    Path(a.out, "pg_filtered_recall.json").write_text(json.dumps(rows, indent=2))
    meas = [r for r in rows if "recall_vs_exact@10" in r]
    cols = ["tenants", "chunks", "mode", "ef_search", "recall_vs_exact@10", "empty_results", "p95_ms", "index_MB", "relations", "build_s", "pgvector"]
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in meas]
    md += ["", "Production query plan check:", ""] + [f"- {r['tenants']} tenants: uses HNSW = **{r['uses_hnsw']}** ({r['note']})" for r in rows if r["mode"] == "production_query_plan"]
    Path(a.out, "pg_filtered_recall.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
