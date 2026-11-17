"""Retrieval and generation metrics. Relevance is judged at document level so
golden sets survive re-chunking (they reference doc ids, never chunk ids)."""
from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

import numpy as np


def _dedupe(ranked: Sequence[str]) -> list[str]:
    seen, out = set(), []
    for r in ranked:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return float("nan")
    return len(rel.intersection(_dedupe(ranked)[:k])) / len(rel)


def mrr(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    if not rel:
        return float("nan")
    for i, r in enumerate(_dedupe(ranked), 1):
        if r in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(_dedupe(ranked)[:k], 1) if r in rel)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal if ideal else 0.0


def nanmean(xs: Sequence[float]) -> float:
    a = np.array([x for x in xs if not (isinstance(x, float) and math.isnan(x))], dtype=float)
    return float(a.mean()) if a.size else float("nan")


def percentile(xs: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs, dtype=float), q)) if len(xs) else float("nan")


# ---- generation metrics (deterministic proxies; swap for an LLM judge in production) ----

_SENT = re.compile(r"(?<=[.!?])\s+")


def citation_accuracy(answer_cites: Sequence[str], context_doc_ids: Iterable[str]) -> float:
    """Fraction of cited doc ids that were actually in the retrieved context."""
    cites = list(answer_cites)
    if not cites:
        return float("nan")
    ctx = set(context_doc_ids)
    return sum(c in ctx for c in cites) / len(cites)


def answer_supported(answer: str, context: str) -> float:
    """Token-overlap faithfulness proxy: share of answer content tokens present in context."""
    from tenantrag.text import tokenize

    a = [t for t in tokenize(answer) if len(t) > 2]
    if not a:
        return float("nan")
    c = set(tokenize(context))
    return sum(t in c for t in a) / len(a)


def exact_answer_hit(answer: str, gold: str | None) -> float:
    if gold is None:
        return float("nan")
    return 1.0 if gold.lower() in answer.lower() else 0.0
