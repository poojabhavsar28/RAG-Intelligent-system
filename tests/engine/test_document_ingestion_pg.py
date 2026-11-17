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
