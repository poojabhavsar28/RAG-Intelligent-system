"""Real PostgreSQL repository integration tests using a restricted runtime role."""
from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.infrastructure.postgres.repositories import (
    PostgresChatRepository,
    PostgresQATemplateRepository,
    PostgresSessionRepository,
    PostgresTenantRepository,
)
from app.infrastructure.postgres.tenant_db import tenant_transaction


@pytest.mark.skipif(
    not os.getenv("POSTGRES_DATABASE_URL"),
    reason="POSTGRES_DATABASE_URL is required for PostgreSQL integration tests",
)
def test_real_repositories_authentication_history_and_tenant_context():
    admin_url = os.environ["POSTGRES_DATABASE_URL"]
    admin_engine = create_engine(
        admin_url.replace("postgresql://", "postgresql+psycopg://", 1)
    )
    runtime_role = "tenantrag_repo_runtime_test"
    runtime_password = "ci-repo-runtime-password"
    tenant_a = f"repo-it-a-{uuid4()}"
    tenant_b = f"repo-it-b-{uuid4()}"
    session_id = str(uuid4())
    session_hash = f"session-hash-{uuid4()}"
    customer_hash = f"customer-hash-{uuid4()}"
    tenant_key_hash = f"tenant-key-hash-{uuid4()}"

    runtime_engine = None
    try:
        with admin_engine.begin() as conn:
            conn.execute(text(
                "DO $$ BEGIN CREATE ROLE tenantrag_repo_runtime_test "
                "LOGIN PASSWORD 'ci-repo-runtime-password' NOSUPERUSER NOBYPASSRLS; "
                "EXCEPTION WHEN duplicate_object THEN "
                "ALTER ROLE tenantrag_repo_runtime_test WITH LOGIN "
                "PASSWORD 'ci-repo-runtime-password' NOSUPERUSER NOBYPASSRLS; "
                "END $$"
            ))
            conn.execute(text("GRANT USAGE ON SCHEMA public TO tenantrag_repo_runtime_test"))
            conn.execute(text(
                "GRANT SELECT, INSERT, UPDATE, DELETE ON tenants, app_users, sessions, "
                "customer_tokens, chat_history, qa_templates, documents, document_chunks, "
                "document_acl TO tenantrag_repo_runtime_test"
            ))
            conn.execute(text(
                "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public "
                "TO tenantrag_repo_runtime_test"
            ))
            for signature in (
                "public.lookup_auth_token(TEXT, TEXT, UUID)",
                "public.lookup_session_tenant(UUID)",
                "public.register_auth_token(TEXT, TEXT, TEXT, TEXT, UUID, TIMESTAMPTZ)",
            ):
                conn.execute(text(f"GRANT EXECUTE ON FUNCTION {signature} TO tenantrag_repo_runtime_test"))
            conn.execute(
                text("INSERT INTO tenants (tenant_id, name) VALUES (:a, 'Repo Test A'), (:b, 'Repo Test B')"),
                {"a": tenant_a, "b": tenant_b},
            )

        runtime_url = make_url(admin_url).set(
            drivername="postgresql+psycopg",
            username=runtime_role,
            password=runtime_password,
        )
        runtime_engine = create_engine(runtime_url, pool_size=1, max_overflow=0)

        # Seed a tenant API key through the narrowly scoped SECURITY DEFINER function.
        with admin_engine.begin() as conn:
            conn.execute(text("""
                SELECT public.register_auth_token(
                    :token_hash, 'tenant_api_key', :tenant_id, :subject_id, NULL, NULL
                )
            """), {
                "token_hash": tenant_key_hash,
                "tenant_id": tenant_a,
                "subject_id": tenant_a,
            })

        tenant_repo = PostgresTenantRepository(runtime_engine)
        session_repo = PostgresSessionRepository(runtime_engine)
        chat_repo = PostgresChatRepository(runtime_engine)
        qa_repo = PostgresQATemplateRepository(runtime_engine)

        assert tenant_repo.resolve_api_key(tenant_key_hash) == tenant_a
        assert tenant_repo.resolve_api_key(f"unknown-{uuid4()}") is None

        session_repo.create_bundle(
            session_id=session_id,
            tenant_id=tenant_a,
            customer_id="customer-1",
            session_token_hash=session_hash,
            customer_token_hash=customer_hash,
            session_ttl_hours=1,
            customer_token_ttl_days=1,
        )
        authenticated_session = session_repo.get_by_id_and_token(session_id, session_hash)
        assert authenticated_session is not None
        assert authenticated_session.tenant_id == tenant_a
        assert session_repo.get_by_id_and_token(session_id, f"wrong-{uuid4()}") is None

        # Session tenant resolution and chat-history retrieval must use the
        # SECURITY DEFINER lookup, then apply RLS in the tenant transaction.
        assert session_repo.get_by_id(session_id).tenant_id == tenant_a
        chat_repo.append_exchange(session_id, tenant_a, "hello", "hello back")
        history = chat_repo.get_history(session_id)
        assert [item.message for item in history] == ["hello", "hello back"]

        with admin_engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO qa_templates (tenant_id, language, question, answer)
                VALUES (:tenant_id, 'en', 'private question', 'private answer')
            """), {"tenant_id": tenant_a})

        assert [item["question"] for item in qa_repo.get_questions(tenant_a, "en")] == [
            "private question"
        ]
        assert qa_repo.get_questions(tenant_b, "en") == []

        # A runtime connection cannot directly inspect the global token index.
        with pytest.raises(Exception):
            with runtime_engine.connect() as conn:
                conn.execute(text("SELECT count(*) FROM auth_token_lookup")).scalar_one()

        # The engine has a one-connection pool: after a tenant transaction ends,
        # the next checkout must not inherit the previous tenant's local setting.
        with tenant_transaction(runtime_engine, tenant_a) as conn:
            assert conn.execute(text(
                "SELECT current_setting('app.tenant_id', true)"
            )).scalar_one() == tenant_a
        with runtime_engine.connect() as conn:
            assert conn.execute(text(
                "SELECT current_setting('app.tenant_id', true)"
            )).scalar_one() in (None, "")

        # RLS must reject a write that explicitly attempts to cross tenant boundaries.
        with pytest.raises(Exception):
            with tenant_transaction(runtime_engine, tenant_a) as conn:
                conn.execute(text("""
                    INSERT INTO qa_templates (tenant_id, language, question, answer)
                    VALUES (:tenant_id, 'en', 'spoofed', 'must fail')
                """), {"tenant_id": tenant_b})

    finally:
        if runtime_engine is not None:
            runtime_engine.dispose()
        with admin_engine.begin() as conn:
            conn.execute(text("DELETE FROM tenants WHERE tenant_id IN (:a, :b)"),
                         {"a": tenant_a, "b": tenant_b})
        admin_engine.dispose()
