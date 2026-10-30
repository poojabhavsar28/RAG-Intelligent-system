"""Tenant-scoped PostgreSQL transaction helpers.

Tenant identifiers are opaque strings inherited from the existing application.
They are never parsed, normalized, or regenerated as UUIDs. Call this helper
only after authentication has resolved the tenant from trusted server identity.
SET LOCAL scopes the tenant setting to this transaction, preventing pooled
connections from carrying one request's tenant context into another.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import Connection


def validate_tenant_id(tenant_id: str) -> str:
    """Validate a tenant identifier without changing its exact value."""
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    return tenant_id


def create_postgres_engine(
    database_url: str,
    *,
    pool_size: int = 10,
    max_overflow: int = 10,
    pool_timeout: int = 30,
) -> Engine:
    """Create a synchronous psycopg-backed engine with bounded pooling."""
    if not database_url:
        raise ValueError("POSTGRES_DATABASE_URL must be configured")
    normalized_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(
        normalized_url,
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        future=True,
    )


@contextmanager
def tenant_transaction(engine: Engine, tenant_id: str) -> Iterator[Connection]:
    """Open a transaction with PostgreSQL RLS tenant context set locally."""
    trusted_tenant_id = validate_tenant_id(tenant_id)

    with engine.begin() as connection:
        connection.execute(
            text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
            {"tenant_id": trusted_tenant_id},
        )
        yield connection
