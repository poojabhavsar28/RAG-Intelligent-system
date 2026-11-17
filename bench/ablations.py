"""Ablations: (A) each retrieval component, (B) the self-correcting loop.

    python bench/ablations.py

(A) dense-only, BM25-only, hybrid RRF, hybrid + rerank on the same golden set.
(B) no self-check / abstain-only / repair+retry on answerable (incl. typo'd), UNANSWERABLE
    (near-miss subject: a sibling exists with a different digit) and FORBIDDEN (other users' docs)
    questions. Hallucination = a non-abstaining answer to an unanswerable or forbidden question.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tenantrag.corpus.simulator import QUERY_TEMPLATES, GoldenItem, build_corpus
from tenantrag.doc_store import InMemoryDocStore
from tenantrag.eval.harness import evaluate_retrieval
from tenantrag.eval.metrics import percentile
from tenantrag.pipeline.llm import ExtractiveLLM
from tenantrag.pipeline.orchestrator import PipelineConfig, RagPipeline
from tenantrag.retrieval.hybrid import HybridRetriever
from tenantrag.retrieval.rerank import OverlapReranker
from tenantrag.tenancy.router import TenantIndexRouter
from tenantrag.text import HashingEmbedder


def md(rows):
    cols = list(rows[0])
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
                     + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows])


def unanswerable(corpus, per_tenant: int, rng: random.Random) -> list[GoldenItem]:
    out = []
    for tid in corpus.tenants:
        docs = [d for d in corpus.docs.values() if d.tenant_id == tid]
        have = {d.subject for d in docs}
        for d in rng.sample(docs, min(per_tenant, len(docs))):
            base = d.subject.split("-")[0]
            free = [x for x in range(1, 10) if f"{base}-{x}" not in have]
            if free and corpus.users[tid]:
                subj = f"{base}-{rng.choice(free)}"
                q = rng.choice(QUERY_TEMPLATES).format(attr=d.attr, subject=subj)
                out.append(GoldenItem(tid, corpus.users[tid][0], q, (), None, "unanswerable"))
    return out


def main():
    corpus = build_corpus(20, max_docs=300, min_docs=6, queries_per_tenant=8, seed=11)
    emb = HashingEmbedder(128)
    router = TenantIndexRouter("dedicated", emb)
    router.ingest(corpus.chunks)
    ds = InMemoryDocStore.from_corpus(corpus)
    retr = HybridRetriever(router, emb, reranker=OverlapReranker(), acl_authority=ds.authority)

    # ---- A: retrieval components ----
    A = []
    for name, kw in [("dense only", dict(lexical=False, rerank=False)), ("BM25 only", dict(dense=False, rerank=False)),
                     ("hybrid (RRF)", dict(rerank=False)), ("hybrid + rerank", {})]:
        r = evaluate_retrieval(corpus.golden, lambda g: retr.retrieve(g.principal, g.query, 10, **kw)).as_row()
        A.append({"config": name, "recall@1": r["recall_at_1"], "recall@5": r["recall_at_5"], "MRR": r["mrr"],
                  "nDCG@10": r["ndcg_at_10"], "p95_ms": r["p95_ms"], "leaks": r["leaks"]})

    # ---- B: self-correcting loop ----
    rng = random.Random(5)
    unans = unanswerable(corpus, 5, rng)
    ans = [g for g in corpus.golden if g.kind == "answerable"]
    forb = [g for g in corpus.golden if g.kind == "forbidden"]
    B = []
    for name, cfg in [("no self-check", PipelineConfig(self_check=False)),
                      ("judge + abstain (1 attempt)", PipelineConfig(self_check=True, max_attempts=1)),
                      ("judge + repair + retry (2 attempts)", PipelineConfig(self_check=True, max_attempts=2))]:
        pipe = RagPipeline(retr, ExtractiveLLM(), config=cfg)
        lat, correct, abst_ans, hall_u, ans_f, leak_f = [], 0, 0, 0, 0, 0
        for g in ans:
            t = time.perf_counter()
            x = pipe.answer(g.principal, g.query)
            lat.append((time.perf_counter() - t) * 1e3)
            correct += bool(g.answer and g.answer in x.answer)
            abst_ans += x.abstained
        for g in unans:
            x = pipe.answer(g.principal, g.query)
            hall_u += not x.abstained
        for g in forb:
            x = pipe.answer(g.principal, g.query)
            fd = corpus.docs[g.forbidden_doc_id]
            ans_f += not x.abstained
            leak_f += (g.forbidden_doc_id in x.citations) or (fd.value in x.answer and fd.subject in x.answer)
        B.append({"config": name, "answer_accuracy": round(correct / len(ans), 3),
                  "wrongly_abstained": round(abst_ans / len(ans), 3),
                  "hallucination_unanswerable": round(hall_u / max(1, len(unans)), 3),
                  "answered_forbidden_q(near-miss)": round(ans_f / max(1, len(forb)), 3),
                  "leaked_forbidden_doc": leak_f,
                  "p50_ms": round(percentile(lat, 50), 2), "p95_ms": round(percentile(lat, 95), 2),
                  "n_ans/unans/forb": f"{len(ans)}/{len(unans)}/{len(forb)}"})
    Path("results").mkdir(exist_ok=True)
    Path("results/ablations.json").write_text(json.dumps({"retrieval": A, "self_check": B}, indent=2))
    Path("results/ablations.md").write_text("### Retrieval components\n\n" + md(A) + "\n\n### Self-correcting loop\n\n" + md(B) + "\n")
    print(md(A)); print(); print(md(B))


if __name__ == "__main__":
    main()
