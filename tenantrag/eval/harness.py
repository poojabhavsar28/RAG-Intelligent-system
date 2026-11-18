"""Evaluation harness: runs golden sets through any retriever configuration.

Evaluation is first-class: every ablation, tenancy mode and regression gate
goes through ``evaluate_retrieval`` so numbers are comparable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from tenantrag.corpus.simulator import GoldenItem
from tenantrag.eval.metrics import mrr, ndcg_at_k, nanmean, percentile, recall_at_k
from tenantrag.retrieval.hybrid import RetrievalResult

RetrieveFn = Callable[[GoldenItem], RetrievalResult]


@dataclass
class RetrievalReport:
    n: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    p50_ms: float
    p95_ms: float
    leaks: int  # forbidden documents returned, summed over forbidden probes (target: 0)
    tenant_violations: int
    per_tenant_recall5: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "per_tenant_recall5"}
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def evaluate_retrieval(golden: Sequence[GoldenItem], retrieve: RetrieveFn, k: int = 10) -> RetrievalReport:
    r1, r5, r10, rr, nd, lat = [], [], [], [], [], []
    per_t: dict[str, list[float]] = {}
    leaks = viol = 0
    for g in golden:
        t0 = time.perf_counter()
        res = retrieve(g)
        lat.append((time.perf_counter() - t0) * 1e3)
        ranked = [h.doc_id for h in res.hits]
        viol += res.tenant_violations
        if g.kind == "forbidden":
            leaks += sum(1 for d in ranked if d == g.forbidden_doc_id)
            continue
        r1.append(recall_at_k(ranked, g.relevant_doc_ids, 1))
        r5.append(recall_at_k(ranked, g.relevant_doc_ids, 5))
        r10.append(recall_at_k(ranked, g.relevant_doc_ids, 10))
        rr.append(mrr(ranked, g.relevant_doc_ids))
        nd.append(ndcg_at_k(ranked, g.relevant_doc_ids, 10))
        per_t.setdefault(g.tenant_id, []).append(r5[-1])
    return RetrievalReport(
        n=len(r5),
        recall_at_1=nanmean(r1), recall_at_5=nanmean(r5), recall_at_10=nanmean(r10),
        mrr=nanmean(rr), ndcg_at_10=nanmean(nd),
        p50_ms=percentile(lat, 50), p95_ms=percentile(lat, 95),
        leaks=leaks, tenant_violations=viol,
        per_tenant_recall5={t: nanmean(v) for t, v in per_t.items()},
    )
