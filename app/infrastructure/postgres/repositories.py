"""PostgreSQL implementations of the existing repository contracts.

The auth-token index is queried only to resolve trusted tenant context before
RLS is enabled for the rest of a request. All tenant-owned reads and writes
run inside tenant_transaction().
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import Engine, text

from app.infrastructure.postgres.tenant_db import tenant_transaction


@dataclass
class SessionRecord:
    session_id: str
    tenant_id: str
    customer_id: str
    language: str = "en"


@dataclass
class TokenRecord:
    tenant_id: str
    customer_id: str


@dataclass
class ChatMessageRecord:
    role: str
    message: str
    sender: str
    timestamp: datetime


class PostgresTenantRepository:
    def __init__(self, engine: Engine):
        self._engine = engine

    def get_by_id(self, tenant_id: str) -> Optional[dict]:
        with tenant_transaction(self._engine, tenant_id) as conn:
            row = conn.execute(
                text("SELECT tenant_id, name, status, plan FROM tenants WHERE tenant_id = :tenant_id"),
                {"tenant_id": tenant_id},
            ).mappings().first()
            return dict(row) if row else None

    def exists(self, tenant_id: str) -> bool:
        return self.get_by_id(tenant_id) is not None

    def resolve_api_key(self, token_hash: str) -> Optional[str]:
        """Resolve a tenant from a hashed tenant API key before RLS context exists."""
        with self._engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT tenant_id FROM auth_token_lookup
                    WHERE token_hash = :token_hash
                      AND token_type = 'tenant_api_key'
                      AND revoked_at IS NULL
                      AND (expires_at IS NULL OR expires_at > now())
                """),
                {"token_hash": token_hash},
            ).mappings().first()
            return row["tenant_id"] if row else None


class PostgresSessionRepository:
    def __init__(self, engine: Engine):
        self._engine = engine

    def create(
        self,
        session_id: str,
        tenant_id: str,
        customer_id: str,
        api_token_hash: str,
        ttl_hours: int,
        language: str = "en",
    ) -> None:
        sid = UUID(str(session_id))
        expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
        with tenant_transaction(self._engine, tenant_id) as conn:
            conn.execute(
                text("""
                    INSERT INTO sessions
                        (session_id, tenant_id, customer_id, token_hash, language, expires_at)
                    VALUES
                        (:session_id, :tenant_id, :customer_id, :token_hash, :language, :expires_at)
                """),
                {
                    "session_id": sid,
                    "tenant_id": tenant_id,
                    "customer_id": str(customer_id),
                    "token_hash": api_token_hash,
                    "language": language,
                    "expires_at": expires_at,
                },
            )
            conn.execute(
                text("""
                    INSERT INTO auth_token_lookup
                        (token_hash, token_type, tenant_id, subject_id, session_id, expires_at)
                    VALUES
                        (:token_hash, 'session', :tenant_id, :subject_id, :session_id, :expires_at)
                """),
                {
                    "token_hash": api_token_hash,
                    "tenant_id": tenant_id,
                    "subject_id": str(customer_id),
                    "session_id": sid,
                    "expires_at": expires_at,
                },
            )

    def create_bundle(
        self,
        session_id: str,
        tenant_id: str,
        customer_id: str,
        session_token_hash: str,
        customer_token_hash: str,
        session_ttl_hours: int,
        customer_token_ttl_days: int,
        language: str = "en",
    ) -> None:
        """Create session and both token records atomically within one tenant transaction."""
        sid = UUID(str(session_id))
        now = datetime.now(timezone.utc)
        session_expiry = now + timedelta(hours=session_ttl_hours)
        customer_expiry = now + timedelta(days=customer_token_ttl_days)
        with tenant_transaction(self._engine, tenant_id) as conn:
            conn.execute(
                text("""
                    INSERT INTO sessions
                        (session_id, tenant_id, customer_id, token_hash, language, created_at, expires_at)
                    VALUES
                        (:session_id, :tenant_id, :customer_id, :token_hash, :language, :created_at, :expires_at)
                """),
                {
                    "session_id": sid, "tenant_id": tenant_id,
                    "customer_id": str(customer_id), "token_hash": session_token_hash,
                    "language": language, "created_at": now, "expires_at": session_expiry,
                },
            )
            conn.execute(
                text("""
                    INSERT INTO customer_tokens
                        (token_hash, tenant_id, customer_id, created_at, expires_at)
                    VALUES (:token_hash, :tenant_id, :customer_id, :created_at, :expires_at)
                """),
                {
                    "token_hash": customer_token_hash, "tenant_id": tenant_id,
                    "customer_id": str(customer_id), "created_at": now,
                    "expires_at": customer_expiry,
                },
            )
            conn.execute(
                text("""
                    INSERT INTO auth_token_lookup
                        (token_hash, token_type, tenant_id, subject_id, session_id, expires_at)
                    VALUES
                        (:session_hash, 'session', :tenant_id, :customer_id, :session_id, :session_expiry),
                        (:customer_hash, 'customer', :tenant_id, :customer_id, NULL, :customer_expiry)
                """),
                {
                    "session_hash": session_token_hash,
                    "customer_hash": customer_token_hash,
                    "tenant_id": tenant_id,
                    "customer_id": str(customer_id),
                    "session_id": sid,
                    "session_expiry": session_expiry,
                    "customer_expiry": customer_expiry,
                },
            )

    def get_by_id_and_token(self, session_id: str, api_token_hash: str) -> Optional[SessionRecord]:
        sid = UUID(str(session_id))
        # This hash-only lookup is the bootstrap step required before RLS can
        # be scoped. The supplied session ID is also checked in the predicate.
        with self._engine.connect() as conn:
            identity = conn.execute(
                text("""
                    SELECT tenant_id FROM auth_token_lookup
                    WHERE token_hash = :token_hash
                      AND token_type = 'session'
                      AND session_id = :session_id
                      AND revoked_at IS NULL
                      AND (expires_at IS NULL OR expires_at > now())
                """),
                {"token_hash": api_token_hash, "session_id": sid},
            ).mappings().first()
        if not identity:
            return None
        return self._get_in_tenant(sid, identity["tenant_id"])

    def get_by_id(self, session_id: str) -> Optional[SessionRecord]:
        sid = UUID(str(session_id))
        with self._engine.connect() as conn:
            identity = conn.execute(
                text("""
                    SELECT tenant_id FROM auth_token_lookup
                    WHERE token_type = 'session' AND session_id = :session_id
                    ORDER BY created_at DESC LIMIT 1
                """),
                {"session_id": sid},
            ).mappings().first()
        if not identity:
            return None
        return self._get_in_tenant(sid, identity["tenant_id"])

    def _get_in_tenant(self, session_id: UUID, tenant_id: str) -> Optional[SessionRecord]:
        with tenant_transaction(self._engine, tenant_id) as conn:
            row = conn.execute(
                text("""
                    SELECT session_id, tenant_id, customer_id, language
                    FROM sessions
                    WHERE tenant_id = :tenant_id AND session_id = :session_id
                      AND (expires_at IS NULL OR expires_at > now())
                """),
                {"tenant_id": tenant_id, "session_id": session_id},
            ).mappings().first()
            return (
                SessionRecord(
                    session_id=str(row["session_id"]),
                    tenant_id=row["tenant_id"],
                    customer_id=row["customer_id"],
                    language=row["language"],
                )
                if row else None
            )


class PostgresTokenRepository:
    def __init__(self, engine: Engine):
        self._engine = engine

    def create(self, token_hash: str, tenant_id: str, customer_id: str, ttl_days: int) -> None:
        expires_at = datetime.now(timezone.utc) + timedelta(days=ttl_days)
        with tenant_transaction(self._engine, tenant_id) as conn:
            conn.execute(
                text("""
                    INSERT INTO customer_tokens (token_hash, tenant_id, customer_id, expires_at)
                    VALUES (:token_hash, :tenant_id, :customer_id, :expires_at)
                """),
                {
                    "token_hash": token_hash,
                    "tenant_id": tenant_id,
                    "customer_id": str(customer_id),
                    "expires_at": expires_at,
                },
            )
            conn.execute(
                text("""
                    INSERT INTO auth_token_lookup
                        (token_hash, token_type, tenant_id, subject_id, expires_at)
                    VALUES (:token_hash, 'customer', :tenant_id, :subject_id, :expires_at)
                """),
                {
                    "token_hash": token_hash,
                    "tenant_id": tenant_id,
                    "subject_id": str(customer_id),
                    "expires_at": expires_at,
                },
            )

    def get_valid(self, token_hash: str) -> Optional[TokenRecord]:
        # Resolve the tenant from the minimal global index, then validate the
        # actual customer-token row under that tenant's RLS context.
        with self._engine.connect() as conn:
            identity = conn.execute(
                text("""
                    SELECT tenant_id, subject_id FROM auth_token_lookup
                    WHERE token_hash = :token_hash AND token_type = 'customer'
                      AND revoked_at IS NULL
                      AND (expires_at IS NULL OR expires_at > now())
                """),
                {"token_hash": token_hash},
            ).mappings().first()
        if not identity:
            return None
        tenant_id = identity["tenant_id"]
        with tenant_transaction(self._engine, tenant_id) as conn:
            row = conn.execute(
                text("""
                    SELECT tenant_id, customer_id FROM customer_tokens
                    WHERE token_hash = :token_hash
                      AND (expires_at IS NULL OR expires_at > now())
                """),
                {"token_hash": token_hash},
            ).mappings().first()
            return TokenRecord(**dict(row)) if row else None


class PostgresChatRepository:
    def __init__(self, engine: Engine):
        self._engine = engine

    def append_exchange(self, session_id: str, tenant_id: str, question: str, answer: str) -> None:
        sid = UUID(str(session_id))
        with tenant_transaction(self._engine, tenant_id) as conn:
            conn.execute(
                text("""
                    INSERT INTO chat_history (session_id, tenant_id, role, message, sender)
                    VALUES
                        (:session_id, :tenant_id, 'user', :question, 'user'),
                        (:session_id, :tenant_id, 'assistant', :answer, 'assistant')
                """),
                {
                    "session_id": sid,
                    "tenant_id": tenant_id,
                    "question": question,
                    "answer": answer,
                },
            )

    def get_history(self, session_id: str) -> list[ChatMessageRecord]:
        sid = UUID(str(session_id))
        with self._engine.connect() as conn:
            identity = conn.execute(
                text("""
                    SELECT tenant_id FROM auth_token_lookup
                    WHERE token_type = 'session' AND session_id = :session_id
                    ORDER BY created_at DESC LIMIT 1
                """),
                {"session_id": sid},
            ).mappings().first()
        if not identity:
            return []
        with tenant_transaction(self._engine, identity["tenant_id"]) as conn:
            rows = conn.execute(
                text("""
                    SELECT role, message, sender, created_at AS timestamp
                    FROM chat_history
                    WHERE tenant_id = :tenant_id AND session_id = :session_id
                    ORDER BY created_at ASC
                """),
                {"tenant_id": identity["tenant_id"], "session_id": sid},
            ).mappings().all()
            return [ChatMessageRecord(**dict(row)) for row in rows]
