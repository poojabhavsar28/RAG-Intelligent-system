"""PostgreSQL RLS integration test using a non-superuser runtime role."""
from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.infrastructure.postgres.tenant_db import tenant_transaction


@pytest.mark.skipif(
    not os.getenv("POSTGRES_DATABASE_URL"),
    reason="POSTGRES_DATABASE_URL is required for PostgreSQL integration tests",
)
def test_rls_isolates_tenants_for_non_superuser_runtime_role():
    admin_url = os.environ["POSTGRES_DATABASE_URL"]
    admin_engine = create_engine(
        admin_url.replace("postgresql://", "postgresql+psycopg://", 1)
    )
    runtime_role = "tenantrag_runtime_test"
    runtime_password = "ci-runtime-password"

    tenant_a = f"rls-test-a-{uuid4()}"
    tenant_b = f"rls-test-b-{uuid4()}"
    question_a = f"tenant-a-{uuid4()}"
    question_b = f"tenant-b-{uuid4()}"

    try:
        with admin_engine.begin() as conn:
            conn.execute(text(
                "DO $$ BEGIN CREATE ROLE tenantrag_runtime_test "
                "LOGIN PASSWORD 'ci-runtime-password' NOSUPERUSER NOBYPASSRLS; "
                "EXCEPTION WHEN duplicate_object THEN "
                "ALTER ROLE tenantrag_runtime_test WITH LOGIN "
                "PASSWORD 'ci-runtime-password' NOSUPERUSER NOBYPASSRLS; "
                "END $$"
            ))
            conn.execute(text(
                "GRANT USAGE ON SCHEMA public TO tenantrag_runtime_test"
            ))
            conn.execute(text(
                "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public "
                "TO tenantrag_runtime_test"
            ))
            conn.execute(text(
                "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public "
                "TO tenantrag_runtime_test"
            ))
            conn.execute(
                text("INSERT INTO tenants (tenant_id, name) VALUES (:id, :name)"),
                [{"id": tenant_a, "name": "RLS Test A"}, {"id": tenant_b, "name": "RLS Test B"}],
            )

        runtime_url = make_url(admin_url).set(
            drivername="postgresql+psycopg",
            username=runtime_role,
            password=runtime_password,
        )
        runtime_engine = create_engine(runtime_url)

        with tenant_transaction(runtime_engine, tenant_a) as conn:
            conn.execute(
                text("""
                    INSERT INTO qa_templates (tenant_id, language, question, answer)
                    VALUES (:tenant_id, 'en', :question, 'answer-a')
                """),
                {"tenant_id": tenant_a, "question": question_a},
            )

        with tenant_transaction(runtime_engine, tenant_b) as conn:
            conn.execute(
                text("""
                    INSERT INTO qa_templates (tenant_id, language, question, answer)
                    VALUES (:tenant_id, 'en', :question, 'answer-b')
                """),
                {"tenant_id": tenant_b, "question": question_b},
            )

        with tenant_transaction(runtime_engine, tenant_a) as conn:
            visible = conn.execute(
                text("SELECT question FROM qa_templates ORDER BY question")
            ).scalars().all()
            cross_tenant = conn.execute(
                text("SELECT question FROM qa_templates WHERE tenant_id = :other"),
                {"other": tenant_b},
            ).scalars().all()
        assert visible == [question_a]
        assert cross_tenant == []

        with runtime_engine.connect() as conn:
            unscoped = conn.execute(
                text("SELECT count(*) FROM qa_templates")
            ).scalar_one()
        assert unscoped == 0

        runtime_engine.dispose()
    finally:
        with admin_engine.begin() as conn:
            conn.execute(text("DELETE FROM tenants WHERE tenant_id IN (:a, :b)"),
                         {"a": tenant_a, "b": tenant_b})
        admin_engine.dispose()
