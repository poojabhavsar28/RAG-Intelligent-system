"""RagPipeline: gateway-free core of a request.

  cache lookup -> understand/route -> retrieve -> judge -> (repair + retry | abstain)
  -> fenced generation with citations -> output guardrails -> cache admission -> response + trace

Every stage is optional and flag-controlled so each one can be ablated and costed."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from tenantrag.eval.metrics import answer_supported
from tenantrag.observability.cost import CostLedger, approx_tokens
from tenantrag.pipeline.llm import LLM
from tenantrag.pipeline.selfcheck import judge, repair_query
from tenantrag.pipeline.understanding import understand
from tenantrag.retrieval.hybrid import HybridRetriever
from tenantrag.security.guards import check_output, extract_citations, fence_context, is_suspect, redact_pii
from tenantrag.types import Hit, Principal

SYSTEM = (
    "Answer ONLY from the provided documents. Cite every claim as [d1], [d2]. "
    "If the documents do not contain the answer, say you don't know."
)
IDK = "I don't know based on the documents available to you."
BLOCKED = "I can't provide that response."


@dataclass
class PipelineConfig:
    top_k: int = 5
    context_k: int = 3
    self_check: bool = True
    max_attempts: int = 2
    abstain_when_insufficient: bool = True
    rerank: bool = True
    guardrails: bool = True
    exclude_suspect_chunks: bool = True


@dataclass
class PipelineResponse:
    answer: str
    citations: list[str] = field(default_factory=list)  # doc ids
    abstained: bool = False
    abstain_reason: Optional[str] = None
    route: str = "retrieve"
    attempts: int = 0
    context_doc_ids: list[str] = field(default_factory=list)
    cache: str = "miss"  # internal only; never serialised to clients
    blocked: bool = False
    reasons: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    support_rate: float = 1.0


class RagPipeline:
    def __init__(self, retriever: HybridRetriever, llm: LLM, *, config: PipelineConfig | None = None,
                 cache=None, ledger: CostLedger | None = None, canaries: tuple[str, ...] = ()):
        self.retriever, self.llm = retriever, llm
        self.cfg = config or PipelineConfig()
        self.cache, self.ledger, self.canaries = cache, ledger or CostLedger(), canaries

    def answer(self, p: Principal, query: str, history: list[str] | None = None) -> PipelineResponse:
        t0 = time.perf_counter()
        plan = understand(query, history)
        if self.cache is not None and not plan.history_dependent and plan.route == "retrieve":
            hit = self.cache.get(p, plan.standalone)
            if hit is not None:
                hit.timings_ms["total"] = (time.perf_counter() - t0) * 1e3
                self.ledger.record(p.tenant_id, 0, 0, 0, hit.cache, (hit.prompt_tokens, hit.completion_tokens))
                return hit
        if plan.route == "direct":
            r = PipelineResponse("Hello! Ask me a question about your documents.", route="direct")
            r.timings_ms["total"] = (time.perf_counter() - t0) * 1e3
            self.ledger.record(p.tenant_id, 0, 0, 0)
            return r

        q, hits, attempts, verdict = plan.standalone, [], 0, None
        for attempts in range(1, (self.cfg.max_attempts if self.cfg.self_check else 1) + 1):
            hits = self.retriever.retrieve(p, q, self.cfg.top_k * 2, rerank=self.cfg.rerank).hits
            if not self.cfg.self_check:
                break
            verdict = judge(q, hits, self.cfg.context_k)
            if verdict.sufficient:
                break
            fixed = repair_query(q, hits, verdict.missing)
            if fixed is None:
                break
            q = fixed
        resp = PipelineResponse("", route="retrieve", attempts=attempts)
        resp.timings_ms["retrieve"] = (time.perf_counter() - t0) * 1e3
        ctx_hits = self._select_context(hits)
        if self.cfg.self_check and self.cfg.abstain_when_insufficient and (verdict is None or not verdict.sufficient):
            resp.answer, resp.abstained = IDK, True
            resp.abstain_reason = f"insufficient_context(coverage={verdict.coverage:.2f})" if verdict else "no_context"
            return self._finish(p, resp, t0, 0, 0, plan, ctx_hits, cache_ok=False)

        labels = {f"d{i}": h for i, h in enumerate(ctx_hits, 1)}
        prompt = fence_context([(lab, h.chunk.text) for lab, h in labels.items()]) + f"\nQUESTION: {plan.standalone}"
        tg = time.perf_counter()
        raw = self.llm.complete(SYSTEM, prompt)
        resp.timings_ms["generate"] = (time.perf_counter() - tg) * 1e3
        pt, ct = approx_tokens(SYSTEM + prompt), approx_tokens(raw)
        resp.context_doc_ids = [h.doc_id for h in labels.values()]
        resp.citations = [labels[c].doc_id for c in dict.fromkeys(extract_citations(raw)) if c in labels]
        resp.answer = raw
        resp.support_rate = answer_supported(raw, " ".join(h.chunk.text for h in labels.values()))
        if "don't know" in raw.lower():
            resp.abstained, resp.abstain_reason = True, "model_abstained"
        if self.cfg.guardrails:
            chk = check_output(raw, labels.keys(), canaries=self.canaries, require_citation=not resp.abstained)
            if not chk.ok:
                resp.reasons = chk.reasons
                if any(r.startswith("citation_not_in_context") or r in ("canary_leak", "pii_in_output") for r in chk.reasons):
                    resp.answer, resp.blocked = BLOCKED, True
                    resp.citations = []
        return self._finish(p, resp, t0, pt, ct, plan, ctx_hits, cache_ok=not (resp.abstained or resp.blocked))

    def _select_context(self, hits: list[Hit]) -> list[Hit]:
        out = []
        for h in hits:
            if self.cfg.exclude_suspect_chunks and is_suspect(h.chunk.text):
                continue
            out.append(h)
            if len(out) == self.cfg.context_k:
                break
        return out

    def _finish(self, p, resp, t0, pt, ct, plan, ctx_hits, cache_ok) -> PipelineResponse:
        resp.prompt_tokens, resp.completion_tokens = pt, ct
        resp.usd = self.ledger.record(p.tenant_id, pt, ct, 1)
        resp.timings_ms["total"] = (time.perf_counter() - t0) * 1e3
        if cache_ok and self.cache is not None and not plan.history_dependent:
            self.cache.put(p, plan.standalone, resp, ctx_hits, pii_shown=bool(redact_pii(resp.answer).counts))
        return resp
