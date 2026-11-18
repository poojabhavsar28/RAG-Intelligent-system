"""Per-tenant adaptation: collect feedback, fit a small tenant-specific reranker, fall back to the
shared reranker during cold start, and report the lift against that shared baseline.

The learned model is a logistic regression over a handful of interpretable features (numpy only),
so it trains in milliseconds per tenant and can be inspected. A cross-encoder fine-tune or an
embedding adapter would slot in behind the same ``rerank`` interface."""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tenantrag.retrieval.rerank import OverlapReranker, Reranker
from tenantrag.text import tokenize
from tenantrag.types import Hit

FEATURES = ("shared_score", "shared_prior", "coverage", "digit_match", "trigram_jaccard", "short_chunk", "bias")
_SHARED = OverlapReranker()


def _trigrams(s: str) -> set[str]:
    s = f"  {s.lower()}  "
    return {s[i : i + 3] for i in range(len(s) - 2)}


def featurize(query: str, hits: Sequence[Hit]) -> np.ndarray:
    q_toks = tokenize(query)
    qset = set(q_toks)
    digits = [t for t in q_toks if any(c.isdigit() for c in t)]
    qtri = _trigrams(query)
    # residual learning: the shared reranker's score and rank are inputs, the tenant model learns a correction
    shared = _SHARED.rerank(query, hits)
    s_score = {h.chunk.chunk_id: h.score for h in shared}
    s_rank = {h.chunk.chunk_id: i for i, h in enumerate(shared)}
    rows = []
    for rank, h in enumerate(hits):
        toks = set(tokenize(h.chunk.text))
        cov = len(qset & toks) / len(qset) if qset else 0.0
        dig = (sum(d in toks for d in digits) / len(digits)) if digits else 0.5
        ctri = _trigrams(h.chunk.text)
        jac = len(qtri & ctri) / len(qtri | ctri) if qtri | ctri else 0.0
        cid = h.chunk.chunk_id
        rows.append([s_score[cid], 1.0 / (s_rank[cid] + 1), cov, dig, jac, 1.0 if len(toks) < 30 else 0.0, 1.0])
    return np.asarray(rows, dtype=np.float64)


@dataclass
class Feedback:
    query: str
    doc_id: str
    signal: int  # +1 thumbs up / clicked citation, -1 thumbs down
    clicked: bool = False


class FeedbackStore:
    """Tenant-keyed; a tenant can only read its own feedback."""

    def __init__(self):
        self._by: dict[str, list[Feedback]] = defaultdict(list)
        self._lock = threading.Lock()

    def add(self, tenant_id: str, fb: Feedback) -> None:
        with self._lock:
            self._by[tenant_id].append(fb)

    def for_tenant(self, tenant_id: str) -> list[Feedback]:
        with self._lock:
            return list(self._by[tenant_id])


class TenantReranker:
    name = "tenant-logreg"

    def __init__(self, weights: np.ndarray):
        self.w = weights

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        if not hits:
            return []
        s = featurize(query, hits) @ self.w
        return [Hit(h.chunk, float(x), "rerank") for h, x in sorted(zip(hits, s), key=lambda t: -t[1])]


def fit(examples: list[tuple[np.ndarray, int]], l2: float = 0.05, iters: int = 600, lr: float = 0.3) -> np.ndarray:
    """Logistic regression shrunk toward the SHARED model (weight 1 on the shared score, 0 elsewhere):
    with little or noisy feedback the tenant model stays close to the baseline instead of overfitting."""
    X = np.vstack([x for x, _ in examples])
    y = np.array([lab for _, lab in examples], dtype=np.float64)
    pos_w = (len(y) - y.sum()) / max(1.0, y.sum())  # rebalance 1 positive vs many negatives
    sw = np.where(y > 0, pos_w, 1.0)
    w0 = np.zeros(X.shape[1])
    w0[0] = 3.0
    w = w0.copy()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        w -= lr * (X.T @ (sw * (p - y)) / sw.sum() + l2 * (w - w0))
    return w


class AdaptiveReranker:
    """Routes each request to its tenant's model once enough feedback exists; otherwise the shared one."""

    name = "adaptive"

    def __init__(self, shared: Reranker | None = None, min_examples: int = 20):
        self.shared = shared or OverlapReranker()
        self.min_examples = min_examples
        self._models: dict[str, TenantReranker] = {}

    def is_adapted(self, tenant_id: str) -> bool:
        return tenant_id in self._models

    def train(self, tenant_id: str, store: FeedbackStore, candidates_for) -> bool:
        """``candidates_for(query) -> list[Hit]`` re-runs retrieval so negatives are real candidates."""
        fbs = [f for f in store.for_tenant(tenant_id)]
        positives = [f for f in fbs if f.signal > 0 or f.clicked]
        if len(positives) < self.min_examples:
            return False  # cold start: keep using the shared model
        ex: list[tuple[np.ndarray, int]] = []
        for f in fbs:
            hits = candidates_for(f.query)
            if not hits:
                continue
            F = featurize(f.query, hits)
            pos = f.signal > 0 or f.clicked
            for h, row in zip(hits, F):
                if h.doc_id == f.doc_id:
                    ex.append((row, 1 if pos else 0))
                elif pos:
                    ex.append((row, 0))  # other candidates were not chosen
        if len(ex) < self.min_examples or not any(l for _, l in ex):
            return False
        self._models[tenant_id] = TenantReranker(fit(ex))
        return True

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        if not hits:
            return []
        m = self._models.get(hits[0].chunk.tenant_id)
        return m.rerank(query, hits) if m else self.shared.rerank(query, hits)
