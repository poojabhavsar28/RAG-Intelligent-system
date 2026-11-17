"""FastAPI surface for the TenantRAG pipeline (/v2).

  POST /v2/ask       gateway (rate/quota/in-flight) -> pipeline -> response (no internals exposed)
  POST /v2/feedback  thumbs / clicked citation, stored per tenant
  GET  /v2/usage     the caller's OWN tenant usage and cost
  GET  /healthz

Identity comes from a ``PrincipalResolver`` dependency. ``static_token_resolver`` is for demos and
tests; production wires the existing session/JWT auth (app/api/deps.py) so tenant_id and group
membership come from server-side identity and never from the request body.
"""
from __future__ import annotations

import uuid
from typing import Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from tenantrag.adaptation.tenant_reranker import Feedback, FeedbackStore
from tenantrag.gateway.limits import Gateway
from tenantrag.pipeline.orchestrator import RagPipeline
from tenantrag.types import Principal

PrincipalResolver = Callable[[str | None], Principal]


def static_token_resolver(tokens: dict[str, Principal]) -> PrincipalResolver:
    def resolve(authorization: str | None) -> Principal:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token")
        p = tokens.get(authorization.split(" ", 1)[1].strip())
        if p is None:
            raise HTTPException(401, "invalid token")
        return p

    return resolve


class AskIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    history: list[str] = Field(default_factory=list, max_length=10)


class AskOut(BaseModel):
    request_id: str
    answer: str
    citations: list[str]
    abstained: bool


class FeedbackIn(BaseModel):
    query: str
    document_id: str
    signal: int = Field(ge=-1, le=1)
    clicked: bool = False


def create_app(pipeline: RagPipeline, gateway: Gateway, resolver: PrincipalResolver,
               feedback: FeedbackStore | None = None) -> FastAPI:
    app = FastAPI(title="TenantRAG", version="2.0")
    fb_store = feedback or FeedbackStore()

    def principal(authorization: str | None = Header(default=None)) -> Principal:
        return resolver(authorization)

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.post("/v2/ask", response_model=AskOut)
    def ask(body: AskIn, response: Response, p: Principal = Depends(principal)):
        adm = gateway.admit(p.tenant_id)
        if not adm.ok:
            raise HTTPException(429, adm.reason, headers={"Retry-After": str(max(1, int(adm.retry_after_s + 0.999)))})
        usd = 0.0
        try:
            r = pipeline.answer(p, body.query, body.history)
            usd = r.usd
        finally:
            gateway.release(p.tenant_id, usd)
        # fresh per-request fields; cache state and traces stay internal
        return AskOut(request_id=uuid.uuid4().hex, answer=r.answer, citations=r.citations, abstained=r.abstained)

    @app.post("/v2/feedback", status_code=204)
    def give_feedback(body: FeedbackIn, p: Principal = Depends(principal)):
        fb_store.add(p.tenant_id, Feedback(body.query, body.document_id, body.signal, body.clicked))
        if body.signal < 0 and pipeline.cache is not None:
            pipeline.cache.evict(p, body.query)  # a bad answer must not be served again

    @app.get("/v2/usage")
    def usage(p: Principal = Depends(principal)):
        u = pipeline.ledger.usage(p.tenant_id)
        return {"tenant_id": p.tenant_id, "requests": u.requests, "usd": round(u.usd, 6),
                "cache_hits": u.cache_hits, "saved_usd": round(u.saved_usd, 6)}

    return app
