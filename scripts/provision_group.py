"""Create a tenant group and attach existing tenant users by external IdP subject."""
from __future__ import annotations

import argparse
import os

from sqlalchemy import create_engine, text


def provision(tenant_id: str, name: str, external_subjects: list[str]) -> int:
    if not tenant_id or not name or not external_subjects:
        raise ValueError("tenant_id, group name and at least one user subject are required")
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
                raise RuntimeError("Group provisioning requires a BYPASSRLS admin role")
            users = conn.execute(text("""
                SELECT user_id, external_subject FROM app_users
                WHERE tenant_id=:tenant AND external_subject = ANY(:subjects)
            """), {"tenant": tenant_id, "subjects": external_subjects}).all()
            if len(users) != len(set(external_subjects)):
                raise ValueError("One or more user subjects do not exist in this tenant")
            group_id = conn.execute(text("""
                INSERT INTO tenant_groups (tenant_id, name)
                VALUES (:tenant, :name)
                ON CONFLICT (tenant_id, name) DO UPDATE SET name=EXCLUDED.name
                RETURNING group_id
            """), {"tenant": tenant_id, "name": name}).scalar_one()
            conn.execute(text("""
                INSERT INTO app_user_groups (tenant_id, user_id, group_id)
                VALUES (:tenant, :user_id, :group_id)
                ON CONFLICT DO NOTHING
            """), [
                {"tenant": tenant_id, "user_id": row.user_id, "group_id": group_id}
                for row in users
            ])
        return int(group_id)
    finally:
        engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a tenant group and attach users")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--external-subject", action="append", required=True)
    args = parser.parse_args()
    print(provision(args.tenant_id, args.name, args.external_subject))
