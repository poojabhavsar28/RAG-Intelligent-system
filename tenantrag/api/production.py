"""Production TenantRAG API.

This module is intentionally separate from demo.py. It uses authenticated tenant
API keys, PostgreSQL/pgvector persistence and ACL-filtered retrieval, Redis-backed
admission control, durable feedback/usage records, and local Ollama inference.
"""
from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timezone
from typing import Annotated

import httpx
import redis
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.infrastructure.postgres.tenant_db import tenant_transaction
from tenantrag.observability.cost import CostLedger
from tenantrag.pipeline.llm import OllamaLLM
from tenantrag.pipeline.orchestrator import RagPipeline
from tenantrag.retrieval.rerank import OverlapReranker
from tenantrag.tenancy.pg import PgHybridRetriever, PgTenantStore
from tenantrag.text import OllamaEmbedder
from tenantrag.types import Acl, Principal


class LocalCostLedger(CostLedger):
    """Track usage without pretending local Ollama inference has cloud API charges."""

    @staticmethod
    def price(prompt_tokens: int, completion_tokens: int, embed_calls: int) -> float:
        return 0.0


class RedisGateway:
    """Atomic, shared per-tenant rate, concurrency and UTC-day request quotas."""

    _ADMIT = """
    local tm = redis.call('TIME')
    local now = tonumber(tm[1]) + tonumber(tm[2]) / 1000000
    local rate = tonumber(ARGV[1])
    local burst = tonumber(ARGV[2])
    local max_inflight = tonumber(ARGV[3])
    local daily_quota = tonumber(ARGV[4])
    local tokens = tonumber(redis.call('HGET', KEYS[1], 'tokens')) or burst
    local last = tonumber(redis.call('HGET', KEYS[1], 'last')) or now
    tokens = math.min(burst, tokens + math.max(0, now - last) * rate)
    local inflight = tonumber(redis.call('GET', KEYS[3])) or 0
    local requests = tonumber(redis.call('GET', KEYS[2])) or 0
    if requests >= daily_quota then
      return {0, 'daily_quota', 86400}
    end
    if inflight >= max_inflight then
      return {0, 'too_many_in_flight', 1}
    end
    if tokens < 1 then
      return {0, 'rate_limited', math.ceil((1 - tokens) / rate)}
    end
    tokens = tokens - 1
    redis.call('HSET', KEYS[1], 'tokens', tokens, 'last', now)
    redis.call('EXPIRE', KEYS[1], 120)
    redis.call('INCR', KEYS[2])
    redis.call('EXPIRE', KEYS[2], 90000)
    redis.call('INCR', KEYS[3])
    redis.call('EXPIRE', KEYS[3], 3600)
    return {1, 'ok', 0}
    """
    _RELEASE = """
    local n = redis.call('DECR', KEYS[1])
    if n < 0 then redis.call('SET', KEYS[1], 0) end
    return 1
    """

    def __init__(self, client, *, rate_per_s=5, burst=20, max_inflight=4, daily_quota=10000):
        self.redis = client
        self.rate_per_s = rate_per_s
        self.burst = burst
        self.max_inflight = max_inflight
        self.daily_quota = daily_quota

    @staticmethod
    def _tag(tenant_id: str) -> str:
        return hashlib.sha256(tenant_id.encode()).hexdigest()[:32]

    def admit(self, tenant_id: str):
        tag = self._tag(tenant_id)
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        result = self.redis.eval(
            self._ADMIT, 3,
            f"trag:{{{tag}}}:bucket",
            f"trag:{{{tag}}}:requests:{day}",
            f"trag:{{{tag}}}:inflight",
            self.rate_per_s, self.burst, self.max_inflight, self.daily_quota,
        )
        reason = result[1].decode() if isinstance(result[1], bytes) else str(result[1])
        return bool(int(result[0])), reason, int(result[2]), tag

    def release(self, tag: str) -> None:
        self.redis.eval(self._RELEASE, 1, f"trag:{{{tag}}}:inflight")


class AskIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    history: list[str] = Field(default_factory=list, max_length=10)


class AskOut(BaseModel):
    request_id: str
    answer: str
    citations: list[str]
    abstained: bool


class IngestIn(BaseModel):
    external_id: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=2_000_000)
    source_uri: str | None = Field(default=None, max_length=2048)
    language: str = Field(default="en", min_length=2, max_length=16)
    trust_tier: int = Field(default=1, ge=1, le=3)


class FeedbackIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_id: str = Field(min_length=1, max_length=256)
    signal: int = Field(ge=-1, le=1)
    clicked: bool = False


class PipelineRetriever:
    """Adapt the persistent SQL retriever to the pipeline's common retriever protocol."""

    def __init__(self, backend: PgHybridRetriever):
        self.backend = backend
        self.reranker = OverlapReranker()

    def retrieve(self, principal: Principal, query: str, k: int = 10, *, rerank: bool = True):
        result = self.backend.retrieve(principal, query, max(k, 20))
        if rerank and result.hits:
            result.hits = self.reranker.rerank(query, result.hits)[:k]
        else:
            result.hits = result.hits[:k]
        return result


def resolve_principal(engine: Engine, raw_key: str | None) -> Principal:
    if not raw_key or len(raw_key) < 32:
        raise HTTPException(status_code=401, detail="A valid X-Tenant-API-Key is required")
    token_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT tenant_id, subject_id FROM public.lookup_auth_token "
                    "(:token_hash, 'tenant_api_key', NULL)"
                ),
                {"token_hash": token_hash},
            ).mappings().first()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Authentication service unavailable") from exc
    if not row or row["subject_id"] != row["tenant_id"]:
        raise HTTPException(status_code=401, detail="Invalid or expired tenant API key")
    # Tenant service keys intentionally carry no user/group identity. They can
    # access documents granted to the tenant_member role, never user/group ACLs.
    return Principal(tenant_id=row["tenant_id"], user_id=0, group_ids=())


def create_production_app(
    engine: Engine,
    redis_client,
    embedder: OllamaEmbedder,
    llm: OllamaLLM,
    *,
    daily_quota: int = 10000,
) -> FastAPI:
    store = PgTenantStore(engine, embedder)
    sql_retriever = PgHybridRetriever(engine, embedder, audit=True)
    pipeline = RagPipeline(
        PipelineRetriever(sql_retriever),
        llm,
        ledger=LocalCostLedger(),
    )
    gateway = RedisGateway(redis_client, daily_quota=daily_quota)
    app = FastAPI(title="TenantRAG Production API", version="1.0.0")
    app.state.engine = engine
    app.state.redis = redis_client
    app.state.embedder = embedder
    app.state.llm = llm
    app.state.pipeline = pipeline

    def principal(
        api_key: Annotated[str | None, Header(alias="X-Tenant-API-Key")] = None,
    ) -> Principal:
        return resolve_principal(engine, api_key)

    @app.get("/livez")
    def live():
        return {"status": "alive"}

    @app.get("/healthz")
    def ready():
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            redis_client.ping()
            with httpx.Client(timeout=3.0) as client:
                models = client.get(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/tags")
                models.raise_for_status()
            available = {m.get("name") for m in models.json().get("models", [])}
            llm_model = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b")
            embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
            if not any(x == llm_model or x.startswith(llm_model + ":") for x in available):
                raise RuntimeError(f"Required Ollama model is missing: {llm_model}")
            if not any(x == embedding_model or x.startswith(embedding_model + ":") for x in available):
                raise RuntimeError(f"Required Ollama model is missing: {embedding_model}")
            return {"status": "ready", "database": "ok", "redis": "ok", "ollama": "ok"}
        except Exception as exc:
            raise HTTPException(status_code=503, detail="One or more runtime dependencies are unavailable") from exc

    @app.post("/v1/documents", status_code=201)
    def ingest(body: IngestIn, p: Principal = Depends(principal)):
        try:
            result = store.ingest_document(
                p.tenant_id, body.external_id, body.title, body.content,
                acl=Acl(is_public=True), source_uri=body.source_uri,
                language=body.language, trust_tier=body.trust_tier,
            )
            return result
        except (ValueError, httpx.HTTPError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/ask", response_model=AskOut)
    def ask(body: AskIn, p: Principal = Depends(principal)):
        try:
            admitted, reason, retry_after, tag = gateway.admit(p.tenant_id)
        except redis.RedisError as exc:
            raise HTTPException(status_code=503, detail="Rate-limit service unavailable") from exc
        if not admitted:
            raise HTTPException(429, detail=reason, headers={"Retry-After": str(max(1, retry_after))})
        request_id = uuid.uuid4().hex
        try:
            result = pipeline.answer(p, body.query, body.history)
            with tenant_transaction(engine, p.tenant_id) as conn:
                conn.execute(
                    text(
                        "INSERT INTO usage_ledger "
                        "(tenant_id, request_id, prompt_tokens, completion_tokens, embed_calls, cache, usd) "
                        "VALUES (:tenant, :request, :prompt, :completion, 1, 'miss', 0)"
                    ),
                    {
                        "tenant": p.tenant_id, "request": request_id,
                        "prompt": result.prompt_tokens, "completion": result.completion_tokens,
                    },
                )
            return AskOut(
                request_id=request_id, answer=result.answer,
                citations=result.citations, abstained=result.abstained,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail="Local Ollama inference is unavailable") from exc
        finally:
            try:
                gateway.release(tag)
            except redis.RedisError:
                # Redis failure is surfaced by readiness checks; do not mask a
                # successful database/model response while releasing a slot.
                pass

    @app.post("/v1/feedback", status_code=204)
    def feedback(body: FeedbackIn, p: Principal = Depends(principal)):
        with tenant_transaction(engine, p.tenant_id) as conn:
            exists = conn.execute(
                text(
                    "SELECT 1 FROM documents WHERE tenant_id=:tenant AND external_id=:doc "
                    "AND status='active' AND deleted_at IS NULL"
                ),
                {"tenant": p.tenant_id, "doc": body.document_id},
            ).first()
            if not exists:
                raise HTTPException(status_code=404, detail="Document not found")
            conn.execute(
                text(
                    "INSERT INTO feedback (tenant_id, user_id, query, document_id, signal, clicked) "
                    "VALUES (:tenant, 'tenant-api-key', :query, :doc, :signal, :clicked)"
                ),
                {
                    "tenant": p.tenant_id, "query": body.query, "doc": body.document_id,
                    "signal": body.signal, "clicked": body.clicked,
                },
            )
        if body.signal < 0:
            # Persistent cache is deliberately disabled until its storage and
            # invalidation semantics are implemented against Redis.
            pass
        return Response(status_code=204)

    @app.get("/v1/usage")
    def usage(p: Principal = Depends(principal)):
        with tenant_transaction(engine, p.tenant_id) as conn:
            row = conn.execute(
                text(
                    "SELECT count(*) AS requests, coalesce(sum(prompt_tokens),0) AS prompt_tokens, "
                    "coalesce(sum(completion_tokens),0) AS completion_tokens, "
                    "coalesce(sum(usd),0) AS usd FROM usage_ledger WHERE tenant_id=:tenant"
                ),
                {"tenant": p.tenant_id},
            ).mappings().one()
        return {"tenant_id": p.tenant_id, **dict(row)}

    return app


def build_app() -> FastAPI:
    database_url = os.getenv("POSTGRES_DATABASE_URL")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    if not database_url:
        raise RuntimeError("POSTGRES_DATABASE_URL must point to a non-superuser NOBYPASSRLS runtime login")
    engine = create_engine(
        database_url.replace("postgresql://", "postgresql+psycopg://", 1),
        pool_pre_ping=True,
        pool_size=int(os.getenv("POSTGRES_POOL_SIZE", "10")),
        max_overflow=int(os.getenv("POSTGRES_MAX_OVERFLOW", "10")),
        pool_timeout=int(os.getenv("POSTGRES_POOL_TIMEOUT", "30")),
    )
    redis_client = redis.Redis.from_url(redis_url, decode_responses=False)
    embedder = OllamaEmbedder(
        model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        dim=int(os.getenv("OLLAMA_EMBEDDING_DIM", "768")),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        timeout=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
    )
    llm = OllamaLLM(
        model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        timeout=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
    )
    application = create_production_app(engine, redis_client, embedder, llm)

    @application.on_event("shutdown")
    def shutdown_runtime() -> None:
        embedder.close()
        llm.close()
        redis_client.close()
        engine.dispose()

    return application


if os.getenv("POSTGRES_DATABASE_URL"):
    app = build_app()
else:
    app = FastAPI(title="TenantRAG Production API (unconfigured)")

    @app.on_event("startup")
    async def require_runtime_configuration() -> None:
        raise RuntimeError(
            "POSTGRES_DATABASE_URL is required; configure PostgreSQL, Redis and Ollama before startup"
        )
