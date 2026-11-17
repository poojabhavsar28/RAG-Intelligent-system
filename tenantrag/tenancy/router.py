"""TenantIndexRouter: the three tenancy models as one interface.

  dedicated  one dense index (and BM25 index) per tenant. Isolation is structural:
             a query only ever touches its own tenant's index object.
  shared     one dense index for all tenants; tenant + ACL are filters.
  hybrid     tenants at or above ``big_tenant_chunks`` get dedicated indexes,
             the long tail shares one index.

BM25 is always per tenant (own statistics) in every mode, so the modes differ in
the dense leg, memory and operational cost, which is what the benchmark compares.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from tenantrag.retrieval.ann import build_index
from tenantrag.retrieval.bm25 import BM25Index
from tenantrag.text import Embedder
from tenantrag.types import Chunk, Hit, Principal

MODES = ("dedicated", "shared", "hybrid")


class _Dense:
    def __init__(self, chunks: list[Chunk], vecs: np.ndarray, nprobe: int):
        self.chunks = chunks
        self.index = build_index(vecs, nprobe=nprobe)

    def search(self, p: Principal, q: np.ndarray, k: int, ef: int, mode: str) -> list[Hit]:
        chunks = self.chunks

        def mask(idx: np.ndarray) -> np.ndarray:
            return np.fromiter(
                (chunks[i].tenant_id == p.tenant_id and chunks[i].acl.permits(p) for i in idx),
                dtype=bool,
                count=len(idx),
            )

        idx, sc = self.index.search(q, k, ef=ef, mask=mask, mode=mode)
        return [Hit(chunks[i], float(s), "dense") for i, s in zip(idx, sc)]


class TenantIndexRouter:
    def __init__(
        self,
        mode: str,
        embedder: Embedder,
        *,
        big_tenant_chunks: int = 500,
        ef: int = 64,
        nprobe: int = 8,
        filter_mode: str = "post",
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode, self.embedder = mode, embedder
        self.big = big_tenant_chunks
        self.ef, self.nprobe, self.filter_mode = ef, nprobe, filter_mode
        self._dedicated: dict[str, _Dense] = {}
        self._shared: _Dense | None = None
        self._bm25: dict[str, BM25Index] = {}
        self.n_chunks = 0

    # ---- ingestion ----
    def ingest(self, chunks: Sequence[Chunk], vectors: np.ndarray | None = None) -> None:
        """``vectors`` (row-aligned with ``chunks``) lets benchmarks embed once and reuse."""
        by_t: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            by_t[c.tenant_id].append(c)
        self.n_chunks = len(chunks)
        vec_cache: dict[int, np.ndarray] = {}
        allv = vectors if vectors is not None else self.embedder.embed([c.text for c in chunks])
        for c, v in zip(chunks, allv):
            vec_cache[c.chunk_id] = v
        shared_chunks: list[Chunk] = []
        for t, cs in by_t.items():
            self._bm25[t] = BM25Index(cs)
            dedicated = self.mode == "dedicated" or (self.mode == "hybrid" and len(cs) >= self.big)
            if dedicated:
                self._dedicated[t] = _Dense(cs, np.stack([vec_cache[c.chunk_id] for c in cs]), self.nprobe)
            else:
                shared_chunks.extend(cs)
        if shared_chunks:
            self._shared = _Dense(shared_chunks, np.stack([vec_cache[c.chunk_id] for c in shared_chunks]), self.nprobe)

    # ---- search ----
    def _store(self, tenant_id: str) -> _Dense | None:
        if tenant_id in self._dedicated:
            return self._dedicated[tenant_id]
        return self._shared if self.mode != "dedicated" else None

    def dense(self, p: Principal, qvec: np.ndarray, k: int, ef: int | None = None) -> list[Hit]:
        store = self._store(p.tenant_id)
        return store.search(p, qvec, k, ef or self.ef, self.filter_mode) if store else []

    def lexical(self, p: Principal, query: str, k: int) -> list[Hit]:
        idx = self._bm25.get(p.tenant_id)
        if not idx:
            return []
        hits = idx.search(query, k, allow=lambda c: c.tenant_id == p.tenant_id and c.acl.permits(p))
        return [Hit(c, s, "lexical") for c, s in hits]

    # ---- accounting ----
    def stats(self) -> dict:
        stores = list(self._dedicated.values()) + ([self._shared] if self._shared else [])
        return {
            "mode": self.mode,
            "n_indexes": len(stores),
            "n_dedicated": len(self._dedicated),
            "vector_bytes": sum(s.index.nbytes for s in stores),
            "n_chunks": self.n_chunks,
        }
