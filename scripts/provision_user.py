"""Provision a user identity and issue a tenant-scoped, user-bound API key."""
from __future__ import annotations

import argparse
import hashlib
import os
import secrets
from uuid import UUID

from sqlalchemy import create_engine, text


def provision(tenant_id: str, external_subject: str) -> tuple[str, str]:
    if not tenant_id or not external_subject:
        raise ValueError("tenant_id and external_subject are required")
    database_url = os.getenv("POSTGRES_ADMIN_DATABASE_URL")
    if not database_url:
        raise RuntimeError("POSTGRES_ADMIN_DATABASE_URL must be configured")
    engine = create_engine(database_url.replace("postgresql://", "postgresql+psycopg://", 1))
    raw_key = "tru_" + secrets.token_urlsafe(40)
    token_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    try:
        with engine.begin() as conn:
            bypass = conn.execute(text(
                "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname=current_user"
            )).scalar_one()
            if not bypass:
                raise RuntimeError("User provisioning requires a BYPASSRLS admin role")
            user_id = conn.execute(text("""
                INSERT INTO app_users (tenant_id, external_subject)
                VALUES (:tenant, :subject)
                ON CONFLICT (tenant_id, external_subject)
                DO UPDATE SET external_subject=EXCLUDED.external_subject
                RETURNING user_id
            """), {"tenant": tenant_id, "subject": external_subject}).scalar_one()
            conn.execute(text("""
                INSERT INTO auth_token_lookup
                    (token_hash, token_type, tenant_id, subject_id)
                VALUES (:hash, 'tenant_api_key', :tenant, :subject)
            """), {"hash": token_hash, "tenant": tenant_id, "subject": "user:" + str(user_id)})
        return str(user_id), raw_key
    finally:
        engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provision a user API key")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--external-subject", required=True, help="Stable IdP subject, not an email address")
    args = parser.parse_args()
    user_id, key = provision(args.tenant_id, args.external_subject)
    print(f"User ID: {user_id}")
    print("Store this API key securely; it is displayed only once:")
    print(key)
