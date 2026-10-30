"""PostgreSQL RLS integration test using a non-superuser runtime role."""
from __future__ import annotations

import os
from uuid import uuid4

import pytest
from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import make_url

from app.domain.retrieval import RetrievalContext
from app.infrastructure.postgres.tenant_db import tenant_transaction
from app.infrastructure.vectorstore.pgvector_store import PgVectorRetriever


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
    document_a, document_b = uuid4(), uuid4()
    chunk_a, chunk_b = uuid4(), uuid4()

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
                "GRANT SELECT, INSERT, UPDATE, DELETE ON tenants, app_users, sessions, "
                "customer_tokens, chat_history, qa_templates, documents, document_chunks, "
                "document_acl TO tenantrag_runtime_test"
            ))
            conn.execute(text(
                "GRANT EXECUTE ON FUNCTION public.lookup_auth_token(TEXT, TEXT, UUID) "
                "TO tenantrag_runtime_test"
            ))
            conn.execute(text(
                "GRANT EXECUTE ON FUNCTION public.lookup_session_tenant(UUID) "
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

        # Seed one vector per tenant as the migration/admin role, then verify
        # runtime retrieval enforces tenant RLS and document ACLs.
        vector = [1.0] + [0.0] * 767
        with admin_engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO documents
                    (tenant_id, document_id, title, language, checksum, status)
                VALUES
                    (:tenant_id, :document_id, :title, 'en', :checksum, 'active')
            """), [
                {"tenant_id": tenant_a, "document_id": document_a,
                 "title": "A", "checksum": f"checksum-{document_a}"},
                {"tenant_id": tenant_b, "document_id": document_b,
                 "title": "B", "checksum": f"checksum-{document_b}"},
            ])
            chunk_insert = text("""
                INSERT INTO document_chunks
                    (tenant_id, chunk_id, document_id, ordinal, content,
                     content_hash, language, embedding)
                VALUES
                    (:tenant_id, :chunk_id, :document_id, 0, :content,
                     :content_hash, 'en', :embedding)
            """).bindparams(bindparam("embedding", type_=Vector(768)))
            conn.execute(chunk_insert, [
                {"tenant_id": tenant_a, "chunk_id": chunk_a,
                 "document_id": document_a, "content": "private A",
                 "content_hash": f"hash-{chunk_a}", "embedding": vector},
                {"tenant_id": tenant_b, "chunk_id": chunk_b,
                 "document_id": document_b, "content": "private B",
                 "content_hash": f"hash-{chunk_b}", "embedding": vector},
            ])
            conn.execute(text("""
                INSERT INTO document_acl
                    (tenant_id, document_id, principal_type, principal_id, permission)
                VALUES
                    (:tenant_id, :document_id, 'role', 'tenant_member', 'read')
            """), [
                {"tenant_id": tenant_a, "document_id": document_a},
                {"tenant_id": tenant_b, "document_id": document_b},
            ])

        class UnitEmbedder:
            def embed_query(self, query):
                return vector

        retriever = PgVectorRetriever(runtime_engine, UnitEmbedder())
        hits_a = retriever.retrieve(
            RetrievalContext(tenant_id=tenant_a, role_ids=("tenant_member",)),
            "hello",
        )
        assert [hit.text for hit in hits_a] == ["private A"]

        no_acl = retriever.retrieve(
            RetrievalContext(tenant_id=tenant_a),
            "hello",
        )
        assert no_acl == []

        hits_b = retriever.retrieve(
            RetrievalContext(tenant_id=tenant_b, role_ids=("tenant_member",)),
            "hello",
        )
        assert [hit.text for hit in hits_b] == ["private B"]

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
