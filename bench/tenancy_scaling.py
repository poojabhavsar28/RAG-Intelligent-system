"""Tenancy-model scaling benchmark (dedicated vs shared vs hybrid).

    python bench/tenancy_scaling.py --tenants 10 100 1000 --queries 300

Reports, per (tenant count, mode): dense-only recall@5 (isolates the filtered-ANN
effect), hybrid recall@5, p95 latency, vector memory, index count and a MODELED
monthly infrastructure cost per tenant. Cost uses the assumptions below; replace
them with your cloud prices. Isolation leaks must be 0 in every row.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tenantrag.corpus.simulator import build_corpus
from tenantrag.eval.harness import evaluate_retrieval
from tenantrag.retrieval.hybrid import HybridRetriever
from tenantrag.tenancy.router import TenantIndexRouter
from tenantrag.text import HashingEmbedder

# ---- cost model assumptions (editable) ----
USD_PER_GB_MONTH_RAM = 6.0          # memory-resident vector index
PER_INDEX_OVERHEAD_BYTES = 4 << 20  # fixed per-index cost (graph/centroid/metadata), 4 MiB assumption


def monthly_cost_per_tenant(stats: dict, n_tenants: int) -> float:
    total = stats["vector_bytes"] + stats["n_indexes"] * PER_INDEX_OVERHEAD_BYTES
    return total / (1 << 30) * USD_PER_GB_MONTH_RAM / n_tenants


def run(n_tenants: int, n_queries: int, seed: int, ef_sweep: bool) -> list[dict]:
    t0 = time.perf_counter()
    corpus = build_corpus(n_tenants=n_tenants, max_docs=300, min_docs=6, alpha=0.9, queries_per_tenant=6, seed=seed)
    emb = HashingEmbedder(128)
    vecs = emb.embed([c.text for c in corpus.chunks])
    rng = random.Random(seed)
    golden = [g for g in corpus.golden]
    rng.shuffle(golden)
    ans = [g for g in golden if g.kind == "answerable"][: int(n_queries * 0.7)]
    forb = [g for g in golden if g.kind == "forbidden"][: int(n_queries * 0.3)]
    probe = ans + forb
    sizes = sorted((t["n_docs"] * 3 for t in corpus.tenants.values()), reverse=True)
    print(f"[{n_tenants} tenants] {len(corpus.chunks)} chunks, largest tenant {sizes[0]}, median {sizes[len(sizes)//2]} "
          f"(build+embed {time.perf_counter()-t0:.1f}s)", flush=True)
    rows = []
    n = len(corpus.chunks)
    # (mode, ef = candidates fetched before post-filtering, nprobe = IVF cells searched, filter strategy)
    configs = [("dedicated", 64, 8, "post"), ("shared", 64, 8, "post"), ("shared", 64, 8, "pre"), ("hybrid", 64, 8, "post")]
    if ef_sweep:  # raise search effort on the shared index, up to an exact-scan oracle
        configs += [("shared", 256, 16, "post"), ("shared", 1024, 32, "post"), ("shared", n, 10_000, "post"),
                    ("shared", 64, 32, "pre")]
    for mode, ef, nprobe, fm in configs:
        tb = time.perf_counter()
        router = TenantIndexRouter(mode, emb, ef=ef, nprobe=nprobe, filter_mode=fm, big_tenant_chunks=300)
        router.ingest(corpus.chunks, vecs)
        build_s = time.perf_counter() - tb
        hr = HybridRetriever(router, emb, acl_authority=corpus.authoritative_acl, candidates=40)
        dense = evaluate_retrieval(probe, lambda g: hr.retrieve(g.principal, g.query, 10, lexical=False, rerank=False))
        hyb = evaluate_retrieval(probe, lambda g: hr.retrieve(g.principal, g.query, 10, rerank=False))
        st = router.stats()
        rows.append({
            "tenants": n_tenants, "mode": mode, "ef": ef, "nprobe": nprobe, "filter": fm if mode != "dedicated" else "n/a",
            "chunks": len(corpus.chunks),
            "dense_recall@5": round(dense.recall_at_5, 3), "hybrid_recall@5": round(hyb.recall_at_5, 3),
            "dense_p95_ms": round(dense.p95_ms, 2), "hybrid_p95_ms": round(hyb.p95_ms, 2),
            "indexes": st["n_indexes"], "vector_MB": round(st["vector_bytes"] / 1e6, 1),
            "build_s": round(build_s, 1),
            "usd_per_tenant_month": round(monthly_cost_per_tenant(st, n_tenants), 5),
            "leaks": dense.leaks + hyb.leaks, "tenant_violations": dense.tenant_violations + hyb.tenant_violations,
        })
        print(rows[-1], flush=True)
    return rows


def to_markdown(rows: list[dict]) -> str:
    cols = ["tenants", "mode", "filter", "ef", "nprobe", "dense_recall@5", "hybrid_recall@5", "dense_p95_ms", "hybrid_p95_ms",
            "indexes", "vector_MB", "usd_per_tenant_month", "leaks"]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenants", type=int, nargs="+", default=[10, 100, 1000])
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    all_rows = []
    for n in a.tenants:
        all_rows += run(n, a.queries, a.seed, ef_sweep=(n == max(a.tenants)))
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    (out / "tenancy_scaling.json").write_text(json.dumps(all_rows, indent=2))
    (out / "tenancy_scaling.md").write_text(to_markdown(all_rows) + "\n")
    print("\n" + to_markdown(all_rows))
