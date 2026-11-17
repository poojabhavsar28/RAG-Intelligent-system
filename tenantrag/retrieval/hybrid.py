"""Hybrid retrieval: dense + BM25 -> RRF -> authoritative ACL check -> rerank.

ACL semantics (see docs/tenantrag/design.md, "revocation vs grant"):
  * Filtering inside the dense/lexical legs uses the chunk's denormalized ACL copy.
  * The final decision uses the AUTHORITATIVE document ACL. A stale chunk copy
    that still says "allowed" after a revocation is dropped here (revocations are
    enforced immediately). A stale copy that still says "denied" after a grant
    simply delays the grant until propagation (fail-safe direction).
  * Mismatches are counted (``stale_acl_hits``) rather than treated as hard errors.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from tenantrag.retrieval.fusion import reciprocal_rank_fusion
from tenantrag.retrieval.rerank import Reranker
from tenantrag.tenancy.router import TenantIndexRouter
from tenantrag.text import Embedder
from tenantrag.types import Acl, Hit, Principal

AclAuthority = Callable[[str, str], Optional[Acl]]


@dataclass
class RetrievalResult:
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)
    stale_acl_hits: int = 0
    dropped_by_authority: int = 0
    tenant_violations: int = 0  # must always be 0; asserted in tests and the leak suite


class HybridRetriever:
    def __init__(
        self,
        router: TenantIndexRouter,
        embedder: Embedder,
        *,
        reranker: Reranker | None = None,
        acl_authority: AclAuthority | None = None,
        audit=None,
        candidates: int = 40,
        rrf_k: int = 60,
    ):
        self.router, self.embedder = router, embedder
        self.reranker, self.acl_authority, self.audit = reranker, acl_authority, audit
        self.candidates, self.rrf_k = candidates, rrf_k

    def retrieve(
        self,
        p: Principal,
        query: str,
        k: int = 5,
        *,
        dense: bool = True,
        lexical: bool = True,
        rerank: bool = True,
    ) -> RetrievalResult:
        t = {}
        t0 = time.perf_counter()
        legs: list[list[Hit]] = []
        if dense:
            qv = self.embedder.embed_query(query)
            legs.append(self.router.dense(p, qv, self.candidates))
            t["dense"] = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        if lexical:
            legs.append(self.router.lexical(p, query, self.candidates))
            t["lexical"] = (time.perf_counter() - t1) * 1e3
        fused = reciprocal_rank_fusion(legs, self.rrf_k) if len(legs) > 1 else (legs[0] if legs else [])

        res = RetrievalResult([])
        verified: list[Hit] = []
        for h in fused:
            if h.chunk.tenant_id != p.tenant_id:  # defense in depth: cannot happen unless a layer is broken
                res.tenant_violations += 1
                continue
            if self.acl_authority is not None:
                auth = self.acl_authority(h.chunk.tenant_id, h.chunk.doc_id)
                if auth is None or not auth.permits(p):
                    res.dropped_by_authority += 1
                    if h.chunk.acl.permits(p):
                        res.stale_acl_hits += 1
                    continue
            elif not h.chunk.acl.permits(p):
                res.dropped_by_authority += 1
                continue
            verified.append(h)

        t2 = time.perf_counter()
        if rerank and self.reranker and verified:
            verified = self.reranker.rerank(query, verified[: self.candidates])
            t["rerank"] = (time.perf_counter() - t2) * 1e3
        res.hits = verified[:k]
        res.timings_ms = t | {"total": (time.perf_counter() - t0) * 1e3}
        if self.audit is not None:
            self.audit.record(
                p, hashlib.sha256(query.encode()).hexdigest()[:16], [h.chunk.chunk_id for h in res.hits],
                [h.chunk.doc_id for h in res.hits],
            )
        return res
