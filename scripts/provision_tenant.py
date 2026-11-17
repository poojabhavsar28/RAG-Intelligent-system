"""Create a tenant and issue its first tenant-scoped API key.

Run with POSTGRES_ADMIN_DATABASE_URL using a PostgreSQL superuser/BYPASSRLS role.
The plaintext API key is printed once; only its SHA-256 hash is persisted.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, text


def provision(tenant_id: str, name: str, expires_in_days: int | None = None) -> str:
    if not tenant_id or not tenant_id.strip():
        raise ValueError("tenant_id must be non-empty")
    if not name or not name.strip():
        raise ValueError("tenant name must be non-empty")
    if expires_in_days is not None and expires_in_days < 1:
        raise ValueError("expires_in_days must be >= 1")
    database_url = os.getenv("POSTGRES_ADMIN_DATABASE_URL")
    if not database_url:
        raise RuntimeError("POSTGRES_ADMIN_DATABASE_URL must be configured")
    engine = create_engine(
        database_url.replace("postgresql://", "postgresql+psycopg://", 1),
        pool_pre_ping=True,
    )
    raw_key = "tr_" + secrets.token_urlsafe(40)
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
                raise RuntimeError("Tenant provisioning requires a PostgreSQL admin role with BYPASSRLS")
            conn.execute(
                text("""
                    INSERT INTO tenants (tenant_id, name)
                    VALUES (:tenant_id, :name)
                    ON CONFLICT (tenant_id) DO NOTHING
                """),
                {"tenant_id": tenant_id, "name": name},
            )
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
    parser = argparse.ArgumentParser(description="Create a tenant and issue its first API key")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--expires-in-days", type=int)
    args = parser.parse_args()
    key = provision(args.tenant_id, args.name, args.expires_in_days)
    print("Store this key securely; it will not be shown again:")
    print(key)
