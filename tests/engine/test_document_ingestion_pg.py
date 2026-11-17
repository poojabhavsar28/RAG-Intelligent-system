"""Persistent document ingestion integration tests against PostgreSQL/pgvector."""
from __future__ import annotations

import os
from uuid import uuid4

import numpy as np
import pytest
from sqlalchemy import create_engine, text

from app.infrastructure.postgres.tenant_db import create_postgres_engine, tenant_transaction
from tenantrag.tenancy.pg import PgHybridRetriever, PgTenantStore
from tenantrag.types import Acl, Principal

pytestmark = pytest.mark.skipif(
    not os.getenv("POSTGRES_DATABASE_URL"),
    reason="POSTGRES_DATABASE_URL required",
)


class FixedEmbedder:
    dim = 768

    def embed(self, texts):
        rows = np.zeros((len(texts), self.dim), dtype=np.float32)
        if len(texts):
            rows[:, 0] = 1.0
        return rows

    def embed_query(self, query):
        row = np.zeros(self.dim, dtype=np.float32)
        row[0] = 1.0
        return row


def test_ingest_persists_chunks_and_retrieval_enforces_tenant_acl():
    engine = create_postgres_engine(os.environ["POSTGRES_DATABASE_URL"])
    tenant_id = f"ingest-test-{uuid4()}"
    embedder = FixedEmbedder()
    store = PgTenantStore(engine, embedder)
    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO tenants (tenant_id, name) VALUES (:tenant, 'ingestion test')"),
                {"tenant": tenant_id},
            )
        result = store.ingest_document(
            tenant_id,
            "handbook-v1",
            "Employee handbook",
            "The annual leave allowance is twenty days. Requests require manager approval.",
            acl=Acl(is_public=True),
        )
        assert result["status"] == "active"
        assert result["chunks"] >= 1

        principal = Principal(tenant_id=tenant_id, user_id=0, group_ids=())
        hits = PgHybridRetriever(engine, embedder).retrieve(
            principal, "annual leave allowance", 10
        ).hits
        assert hits
        assert all(hit.chunk.tenant_id == tenant_id for hit in hits)
        assert any(hit.doc_id == "handbook-v1" for hit in hits)

        updated = store.ingest_document(
            tenant_id,
            "handbook-v1",
            "Employee handbook v2",
            "The annual leave allowance is twenty-five days.",
            acl=Acl(is_public=True),
        )
        assert updated["status"] == "active"
        with tenant_transaction(engine, tenant_id) as conn:
            rows = conn.execute(
                text("SELECT content FROM document_chunks WHERE tenant_id=:tenant"),
                {"tenant": tenant_id},
            ).scalars().all()
        assert rows
        assert all("twenty-five" in row for row in rows)
        assert all("manager approval" not in row for row in rows)
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM tenants WHERE tenant_id=:tenant"), {"tenant": tenant_id})
        engine.dispose()



def test_private_and_group_documents_are_visible_only_to_authorized_users():
    from tenantrag.tenancy.pg import doc_uuid

    engine = create_postgres_engine(os.environ["POSTGRES_DATABASE_URL"])
    tenant_id = f"acl-api-test-{uuid4()}"
    embedder = FixedEmbedder()
    store = PgTenantStore(engine, embedder)
    user_a, user_b = uuid4(), uuid4()
    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO tenants (tenant_id, name) VALUES (:tenant, 'ACL API test')"),
                {"tenant": tenant_id},
            )
            conn.execute(
                text("""
                    INSERT INTO app_users (tenant_id, user_id, external_subject)
                    VALUES (:tenant, :a, 'subject-a'), (:tenant, :b, 'subject-b')
                """),
                {"tenant": tenant_id, "a": user_a, "b": user_b},
            )
            group_id = conn.execute(
                text("INSERT INTO tenant_groups (tenant_id, name) VALUES (:tenant, 'HR') RETURNING group_id"),
                {"tenant": tenant_id},
            ).scalar_one()
            conn.execute(
                text("INSERT INTO app_user_groups (tenant_id, user_id, group_id) VALUES (:tenant, :user, :group)"),
                {"tenant": tenant_id, "user": user_a, "group": group_id},
            )

        store.ingest_document(
            tenant_id, "private-payroll", "Private payroll",
            "Payroll confidential compensation details.",
            acl=Acl(allow_users=frozenset({str(user_a)})),
            created_by=str(user_a),
        )
        store.ingest_document(
            tenant_id, "hr-handbook", "HR handbook",
            "HR group handbook leave policy.",
            acl=Acl(allow_users=frozenset({str(user_a)}), allow_groups=frozenset({int(group_id)})),
            created_by=str(user_a),
        )

        retriever = PgHybridRetriever(engine, embedder)
        owner = Principal(tenant_id, str(user_a), (int(group_id),))
        other_user = Principal(tenant_id, str(user_b), ())
        tenant_service = Principal(tenant_id, 0, ())
        owner_docs = {
            h.doc_id for h in retriever.retrieve(owner, "payroll confidential compensation", 20).hits
        }
        other_docs = {
            h.doc_id for h in retriever.retrieve(other_user, "payroll confidential compensation", 20).hits
        }
        service_docs = {
            h.doc_id for h in retriever.retrieve(tenant_service, "payroll confidential compensation", 20).hits
        }
        assert "private-payroll" in owner_docs
        assert "private-payroll" not in other_docs
        assert "private-payroll" not in service_docs

        owner_group_docs = {
            h.doc_id for h in retriever.retrieve(owner, "HR group handbook leave policy", 20).hits
        }
        other_group_docs = {
            h.doc_id for h in retriever.retrieve(other_user, "HR group handbook leave policy", 20).hits
        }
        assert "hr-handbook" in owner_group_docs
        assert "hr-handbook" not in other_group_docs
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM tenants WHERE tenant_id=:tenant"), {"tenant": tenant_id})
        engine.dispose()
