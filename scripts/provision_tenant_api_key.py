"""Provision a tenant API key; the plaintext is displayed only once.

Requires POSTGRES_ADMIN_DATABASE_URL with BYPASSRLS. Store the printed key
in the tenant's secret manager and send it as X-Tenant-API-Key.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, text


def provision(tenant_id: str, expires_in_days: int | None = None) -> str:
    if not tenant_id or not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    database_url = os.getenv("POSTGRES_ADMIN_DATABASE_URL")
    if not database_url:
        raise RuntimeError("POSTGRES_ADMIN_DATABASE_URL must be configured")
    engine = create_engine(
        database_url.replace("postgresql://", "postgresql+psycopg://", 1),
        pool_pre_ping=True,
    )
    raw_key = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    expires_at = (
        datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        if expires_in_days is not None else None
    )
    try:
        with engine.begin() as conn:
            can_bypass = conn.execute(text("""
                SELECT r.rolsuper OR r.rolbypassrls
                FROM pg_roles AS r WHERE r.rolname = current_user
            """)).scalar_one()
            if not can_bypass:
                raise RuntimeError("Provisioning requires a PostgreSQL role with BYPASSRLS")
            exists = conn.execute(
                text("SELECT 1 FROM tenants WHERE tenant_id = :tenant_id"),
                {"tenant_id": tenant_id},
            ).first()
            if not exists:
                raise ValueError(f"Unknown tenant_id: {tenant_id!r}")
            conn.execute(text("""
                INSERT INTO auth_token_lookup
                    (token_hash, token_type, tenant_id, subject_id, expires_at)
                VALUES
                    (:token_hash, 'tenant_api_key', :tenant_id, :tenant_id, :expires_at)
            """), {
                "token_hash": token_hash,
                "tenant_id": tenant_id,
                "expires_at": expires_at,
            })
        return raw_key
    finally:
        engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a tenant-scoped API key")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--expires-in-days", type=int)
    args = parser.parse_args()
    key = provision(args.tenant_id, args.expires_in_days)
    print("Store this key securely; it will not be shown again:")
    print(key)
