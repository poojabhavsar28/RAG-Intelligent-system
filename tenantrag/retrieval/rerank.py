"""Rerankers. All implement ``rerank(query, hits) -> hits`` and never add
candidates: they can only reorder what retrieval already authorized."""
from __future__ import annotations

from typing import Protocol, Sequence

from tenantrag.text import tokenize
from tenantrag.types import Hit


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]: ...


class OverlapReranker:
    """Cheap deterministic stand-in: query-term coverage with a rarity weight."""

    name = "overlap"

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        q = set(tokenize(query))
        if not q or not hits:
            return list(hits)
        df: dict[str, int] = {}
        toks = [set(tokenize(h.chunk.text)) for h in hits]
        for s in toks:
            for t in s & q:
                df[t] = df.get(t, 0) + 1
        n = len(hits)
        out = []
        for h, s in zip(hits, toks):
            sc = sum(1.0 + (n - df.get(t, 0)) / n for t in s & q) / len(q)
            out.append(Hit(h.chunk, sc + 1e-3 * h.score, "rerank"))
        return sorted(out, key=lambda h: -h.score)


class CrossEncoderReranker:
    """Hugging Face cross-encoder (e.g. BAAI/bge-reranker-v2-m3 for multilingual)."""

    name = "cross-encoder"

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3", batch_size: int = 16):
        from sentence_transformers import CrossEncoder  # optional heavy dependency

        self._m = CrossEncoder(model)
        self._bs = batch_size

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        if not hits:
            return []
        scores = self._m.predict([(query, h.chunk.text) for h in hits], batch_size=self._bs)
        return sorted((Hit(h.chunk, float(s), "rerank") for h, s in zip(hits, scores)), key=lambda h: -h.score)
