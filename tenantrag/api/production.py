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
from typing import Annotated, Literal

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
    visibility: Literal["tenant", "private", "groups"] = "tenant"
    group_ids: list[int] = Field(default_factory=list, max_length=100)


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
    if not row:
        raise HTTPException(status_code=401, detail="Invalid or expired tenant API key")
    tenant_id = row["tenant_id"]
    subject_id = row["subject_id"]
    try:
        with tenant_transaction(engine, tenant_id) as conn:
            tenant_status = conn.execute(
                text("SELECT status FROM tenants WHERE tenant_id=:tenant"),
                {"tenant": tenant_id},
            ).scalar_one_or_none()
            if tenant_status != "active":
                raise HTTPException(status_code=403, detail="Tenant is not active")
            if subject_id == tenant_id:
                # Tenant service keys can access only tenant-wide documents.
                return Principal(tenant_id=tenant_id, user_id=0, group_ids=())
            if not isinstance(subject_id, str) or not subject_id.startswith("user:"):
                raise HTTPException(status_code=401, detail="Invalid tenant API key subject")
            try:
                user_id = str(uuid.UUID(subject_id.removeprefix("user:")))
            except (ValueError, TypeError, AttributeError) as exc:
                raise HTTPException(status_code=401, detail="Invalid tenant API key subject") from exc
            exists = conn.execute(
                text("SELECT 1 FROM app_users WHERE tenant_id=:tenant AND user_id=:user"),
                {"tenant": tenant_id, "user": user_id},
            ).first()
            if not exists:
                raise HTTPException(status_code=401, detail="User identity is not active in this tenant")
            groups = conn.execute(
                text("SELECT group_id FROM app_user_groups WHERE tenant_id=:tenant AND user_id=:user"),
                {"tenant": tenant_id, "user": user_id},
            ).scalars().all()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Tenant identity service unavailable") from exc
    return Principal(tenant_id=tenant_id, user_id=user_id, group_ids=tuple(int(g) for g in groups))


def create_production_app(
    engine: Engine,
    redis_client,
    embedder: OllamaEmbedder,
    llm: OllamaLLM,
    *,
    daily_quota: int = 10000,
    rate_per_s: float = 5.0,
    burst: int = 20,
    max_inflight: int = 4,
) -> FastAPI:
    store = PgTenantStore(engine, embedder)
    sql_retriever = PgHybridRetriever(engine, embedder, audit=True)
    pipeline = RagPipeline(
        PipelineRetriever(sql_retriever),
        llm,
        ledger=LocalCostLedger(),
    )
    gateway = RedisGateway(
        redis_client, daily_quota=daily_quota, rate_per_s=rate_per_s,
        burst=burst, max_inflight=max_inflight,
    )
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

    def acquire_slot(p: Principal) -> str:
        try:
            admitted, reason, retry_after, tag = gateway.admit(p.tenant_id)
        except redis.RedisError as exc:
            raise HTTPException(status_code=503, detail="Rate-limit service unavailable") from exc
        if not admitted:
            raise HTTPException(429, detail=reason, headers={"Retry-After": str(max(1, retry_after))})
        return tag

    def release_slot(tag: str) -> None:
        try:
            gateway.release(tag)
        except redis.RedisError:
            # Readiness will fail if Redis remains unavailable. Do not mask an
            # already completed database/model operation on slot release.
            pass

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
        tag = acquire_slot(p)
        try:
            if body.visibility == "tenant":
                if body.group_ids:
                    raise HTTPException(status_code=422, detail="group_ids are only valid for groups visibility")
                policy = Acl(is_public=True)
            elif p.user_id == 0:
                raise HTTPException(status_code=403, detail="User-bound API key required for private or group documents")
            elif body.visibility == "private":
                if body.group_ids:
                    raise HTTPException(status_code=422, detail="group_ids are only valid for groups visibility")
                policy = Acl(allow_users=frozenset({p.user_id}))
            else:
                if not body.group_ids or len(set(body.group_ids)) != len(body.group_ids):
                    raise HTTPException(status_code=422, detail="groups visibility requires unique group_ids")
                with tenant_transaction(engine, p.tenant_id) as conn:
                    memberships = conn.execute(
                        text(
                            "SELECT group_id FROM app_user_groups "
                            "WHERE tenant_id=:tenant AND user_id=:user AND group_id=ANY(:groups)"
                        ),
                        {"tenant": p.tenant_id, "user": str(p.user_id), "groups": body.group_ids},
                    ).scalars().all()
                if set(int(g) for g in memberships) != set(body.group_ids):
                    raise HTTPException(status_code=403, detail="You may share only with groups you belong to")
                policy = Acl(
                    allow_users=frozenset({p.user_id}),
                    allow_groups=frozenset(body.group_ids),
                )
            result = store.ingest_document(
                p.tenant_id, body.external_id, body.title, body.content,
                acl=policy, source_uri=body.source_uri,
                language=body.language, trust_tier=body.trust_tier,
                created_by=str(p.user_id) if p.user_id != 0 else "tenant-service",
            )
            return result
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail="Local Ollama embedding service is unavailable") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            release_slot(tag)

    @app.delete("/v1/documents/{external_id}", status_code=204)
    def delete_document(external_id: str, p: Principal = Depends(principal)):
        tag = acquire_slot(p)
        try:
            with tenant_transaction(engine, p.tenant_id) as conn:
                row = conn.execute(
                    text(
                        "SELECT created_by, legal_hold FROM documents "
                        "WHERE tenant_id=:tenant AND external_id=:doc AND status='active' FOR UPDATE"
                    ),
                    {"tenant": p.tenant_id, "doc": external_id},
                ).mappings().first()
                if not row:
                    raise HTTPException(status_code=404, detail="Document not found")
                if p.user_id != 0 and row["created_by"] != str(p.user_id):
                    raise HTTPException(status_code=403, detail="Only the document owner may delete it")
                if row["legal_hold"]:
                    raise HTTPException(status_code=409, detail="Document is under legal hold")
                conn.execute(
                    text(
                        "UPDATE documents SET status='deleted', deleted_at=now(), updated_at=now() "
                        "WHERE tenant_id=:tenant AND external_id=:doc"
                    ),
                    {"tenant": p.tenant_id, "doc": external_id},
                )
            return Response(status_code=204)
        finally:
            release_slot(tag)

    @app.post("/v1/ask", response_model=AskOut)
    def ask(body: AskIn, p: Principal = Depends(principal)):
        tag = acquire_slot(p)
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
            release_slot(tag)

    @app.post("/v1/feedback", status_code=204)
    def feedback(body: FeedbackIn, p: Principal = Depends(principal)):
        tag = acquire_slot(p)
        try:
            with tenant_transaction(engine, p.tenant_id) as conn:
                exists = conn.execute(
                    text(
                        "SELECT 1 FROM documents d WHERE d.tenant_id=:tenant AND d.external_id=:doc "
                        "AND d.status='active' AND d.deleted_at IS NULL "
                        "AND EXISTS (SELECT 1 FROM document_acl a WHERE a.tenant_id=d.tenant_id "
                        "AND a.document_id=d.document_id AND a.permission IN ('read','write','admin') "
                        "AND ((a.principal_type='user' AND a.principal_id=:uid) "
                        "OR (a.principal_type='group' AND a.principal_id=ANY(CAST(:gids AS TEXT[]))) "
                        "OR (a.principal_type='role' AND a.principal_id='tenant_member'))) "
                        "AND NOT EXISTS (SELECT 1 FROM document_acl x WHERE x.tenant_id=d.tenant_id "
                        "AND x.document_id=d.document_id AND x.permission='deny' "
                        "AND ((x.principal_type='user' AND x.principal_id=:uid) "
                        "OR (x.principal_type='group' AND x.principal_id=ANY(CAST(:gids AS TEXT[])))))"

                    ),
                    {
                        "tenant": p.tenant_id, "doc": body.document_id,
                        "uid": str(p.user_id), "gids": [str(g) for g in p.group_ids],
                    },
                ).first()
                if not exists:
                    raise HTTPException(status_code=404, detail="Document not found")
                conn.execute(
                    text(
                        "INSERT INTO feedback (tenant_id, user_id, query, document_id, signal, clicked) "
                        "VALUES (:tenant, :user, :query, :doc, :signal, :clicked)"
                    ),
                    {
                        "tenant": p.tenant_id, "user": str(p.user_id), "query": body.query,
                        "doc": body.document_id, "signal": body.signal, "clicked": body.clicked,
                    },
                )
            return Response(status_code=204)
        finally:
            release_slot(tag)

    @app.get("/v1/usage")
    def usage(p: Principal = Depends(principal)):
        if p.user_id != 0:
            raise HTTPException(status_code=403, detail="Tenant service key required for tenant-wide usage")
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
    application = create_production_app(
        engine, redis_client, embedder, llm,
        daily_quota=int(os.getenv("TENANTRAG_DAILY_REQUEST_QUOTA", "10000")),
        rate_per_s=float(os.getenv("TENANTRAG_RATE_PER_SECOND", "5")),
        burst=int(os.getenv("TENANTRAG_RATE_BURST", "20")),
        max_inflight=int(os.getenv("TENANTRAG_MAX_INFLIGHT", "4")),
    )

    @application.on_event("startup")
    def validate_runtime_role() -> None:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT r.rolsuper, r.rolbypassrls,
                       has_table_privilege(current_user, 'public.documents', 'SELECT') AS can_read_documents,
                       has_table_privilege(current_user, 'public.app_user_groups', 'SELECT') AS can_read_memberships,
                       has_function_privilege(
                           current_user, 'public.lookup_auth_token(text,text,uuid)', 'EXECUTE'
                       ) AS can_lookup_tokens,
                       has_table_privilege(current_user, 'public.auth_token_lookup', 'SELECT') AS can_read_token_table
                FROM pg_roles r WHERE r.rolname = current_user
            """)).mappings().one()
        if row["rolsuper"] or row["rolbypassrls"]:
            raise RuntimeError("TenantRAG API must run with a non-superuser NOBYPASSRLS database role")
        if (
            not row["can_read_documents"] or not row["can_read_memberships"]
            or not row["can_lookup_tokens"] or row["can_read_token_table"]
        ):
            raise RuntimeError("TenantRAG runtime role is missing required least-privilege grants")

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
