"""Per-tenant adaptation experiment.

    python bench/adaptation.py

Per tenant: simulate users (90% click the truly relevant document, 10% mis-click) on a TRAIN query set,
fit the tenant reranker, then evaluate on a held-out TEST set against the shared reranker.
Also shows cold start: with too little feedback the tenant keeps the shared model (lift = 0).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tenantrag.adaptation.tenant_reranker import AdaptiveReranker, Feedback, FeedbackStore
from tenantrag.corpus.simulator import build_corpus
from tenantrag.eval.harness import evaluate_retrieval
from tenantrag.retrieval.hybrid import HybridRetriever
from tenantrag.tenancy.router import TenantIndexRouter
from tenantrag.text import HashingEmbedder


def main(n_tenants: int = 8, seed: int = 4) -> list[dict]:
    corpus = build_corpus(n_tenants, max_docs=250, min_docs=120, queries_per_tenant=60, seed=seed)
    emb = HashingEmbedder(128)
    router = TenantIndexRouter("dedicated", emb)
    router.ingest(corpus.chunks)
    adaptive = AdaptiveReranker(min_examples=20)
    retr = HybridRetriever(router, emb, reranker=adaptive, acl_authority=corpus.authoritative_acl)
    rng = random.Random(seed)
    store = FeedbackStore()
    rows = []
    by_t: dict[str, list] = {}
    for g in corpus.golden:
        if g.kind == "answerable":
            by_t.setdefault(g.tenant_id, []).append(g)
    for t, items in by_t.items():
        half = len(items) // 2
        train, test = items[:half], items[half:]
        cold = train[:5]  # cold start tenant state
        for g in cold:
            store.add(t, Feedback(g.query, g.relevant_doc_ids[0], 1, True))
        base = evaluate_retrieval(test, lambda g: retr.retrieve(g.principal, g.query, 10))
        adaptive.train(t, store, lambda q, p=train[0].principal: retr.retrieve(p, q, 10, rerank=False).hits)
        cold_rep = evaluate_retrieval(test, lambda g: retr.retrieve(g.principal, g.query, 10))
        for g in train:
            doc = g.relevant_doc_ids[0] if rng.random() < 0.9 else rng.choice(list(corpus.docs)[:50])
            store.add(t, Feedback(g.query, doc, 1, True))
        adaptive.train(t, store, lambda q, p=train[0].principal: retr.retrieve(p, q, 10, rerank=False).hits)
        adapted = evaluate_retrieval(test, lambda g: retr.retrieve(g.principal, g.query, 10))
        rows.append({"tenant": t, "feedback": len(store.for_tenant(t)), "test_q": len(test),
                     "shared_r@1": round(base.recall_at_1, 3), "cold_r@1": round(cold_rep.recall_at_1, 3),
                     "adapted_r@1": round(adapted.recall_at_1, 3), "shared_mrr": round(base.mrr, 3),
                     "adapted_mrr": round(adapted.mrr, 3), "lift_mrr": round(adapted.mrr - base.mrr, 3)})
    return rows


if __name__ == "__main__":
    rows = main()
    Path("results").mkdir(exist_ok=True)
    Path("results/adaptation.json").write_text(json.dumps(rows, indent=2))
    cols = list(rows[0])
    avg = {c: round(sum(r[c] for r in rows) / len(rows), 3) for c in cols if c != "tenant"}
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    md.append("| **mean** | " + " | ".join(str(avg[c]) for c in cols if c != "tenant") + " |")
    Path("results/adaptation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
