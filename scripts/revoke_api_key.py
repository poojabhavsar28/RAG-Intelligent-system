"""Revoke a tenant API key using its SHA-256 token hash (never pass plaintext keys)."""
from __future__ import annotations

import argparse
import os
import re

from sqlalchemy import create_engine, text


def revoke(token_hash: str) -> bool:
    if not re.fullmatch(r"[0-9a-f]{64}", token_hash):
        raise ValueError("token_hash must be a lowercase SHA-256 hex digest")
    database_url = os.getenv("POSTGRES_ADMIN_DATABASE_URL")
    if not database_url:
        raise RuntimeError("POSTGRES_ADMIN_DATABASE_URL must be configured")
    engine = create_engine(database_url.replace("postgresql://", "postgresql+psycopg://", 1))
    try:
        with engine.begin() as conn:
            bypass = conn.execute(text(
                "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname=current_user"
            )).scalar_one()
            if not bypass:
                raise RuntimeError("Key revocation requires a BYPASSRLS admin role")
            result = conn.execute(text("""
                UPDATE auth_token_lookup SET revoked_at=now()
                WHERE token_hash=:hash AND revoked_at IS NULL
            """), {"hash": token_hash})
            return result.rowcount == 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Revoke an API key by SHA-256 hash")
    parser.add_argument("--token-hash", required=True)
    args = parser.parse_args()
    print("revoked" if revoke(args.token_hash) else "not found or already revoked")
