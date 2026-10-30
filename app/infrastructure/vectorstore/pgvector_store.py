"""PostgreSQL/pgvector retriever with tenant and document-ACL enforcement."""
from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from pgvector.sqlalchemy import Vector
from sqlalchemy import Engine, bindparam, text

from app.domain.retrieval import Hit, RetrievalContext
from app.infrastructure.postgres.tenant_db import tenant_transaction


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> Sequence[float]:
        """Return a 768-dimensional query embedding for the configured model."""
        ...


class PgVectorRetriever:
    """Cosine-similarity retrieval that always scopes SQL by tenant and ACL."""

    def __init__(self, engine: Engine, embedder: QueryEmbedder, *, dimensions: int = 768):
        if dimensions != 768:
            raise ValueError("The current PostgreSQL schema requires 768-dimensional embeddings")
        self._engine = engine
        self._embedder = embedder
        self._dimensions = dimensions

    def retrieve(
        self,
        context: RetrievalContext,
        query: str,
        k: int = 10,
    ) -> Sequence[Hit]:
        if not query or not query.strip():
            return []
        if not 1 <= k <= 100:
            raise ValueError("k must be between 1 and 100")

        embedding = list(self._embedder.embed_query(query))
        if len(embedding) != self._dimensions:
            raise ValueError(
                f"Expected a {self._dimensions}-dimensional query embedding; got {len(embedding)}"
            )

        statement = text("""
            SELECT
                c.content,
                1 - (c.embedding <=> :embedding) AS score,
                d.metadata || c.metadata AS metadata,
                c.document_id::text AS document_id,
                c.chunk_id::text AS chunk_id
            FROM document_chunks AS c
            JOIN documents AS d
              ON d.tenant_id = c.tenant_id
             AND d.document_id = c.document_id
            WHERE c.tenant_id = :tenant_id
              AND c.language = :language
              AND c.embedding IS NOT NULL
              AND d.status = 'active'
              AND (
                    NOT :require_document_acl
                    OR EXISTS (
                        SELECT 1
                        FROM document_acl AS acl
                        WHERE acl.tenant_id = c.tenant_id
                          AND acl.document_id = c.document_id
                          AND acl.permission IN ('read', 'write', 'admin')
                          AND (
                                (acl.principal_type = 'user'
                                 AND acl.principal_id = :user_id)
                             OR (acl.principal_type = 'group'
                                 AND acl.principal_id = ANY(CAST(:group_ids AS TEXT[])))
                             OR (acl.principal_type = 'role'
                                 AND acl.principal_id = ANY(CAST(:role_ids AS TEXT[])))
                          )
                    )
              )
            ORDER BY c.embedding <=> :embedding
            LIMIT :limit
        """).bindparams(bindparam("embedding", type_=Vector(self._dimensions)))

        params = {
            "embedding": embedding,
            "tenant_id": context.tenant_id,
            "language": context.language,
            "require_document_acl": context.require_document_acl,
            "user_id": context.user_id,
            "group_ids": list(context.group_ids),
            "role_ids": list(context.role_ids),
            "limit": k,
        }

        with tenant_transaction(self._engine, context.tenant_id) as conn:
            rows = conn.execute(statement, params).mappings().all()

        return [
            Hit(
                text=row["content"],
                score=float(row["score"]) if row["score"] is not None else None,
                metadata=row["metadata"] or {},
                document_id=row["document_id"],
                chunk_id=row["chunk_id"],
            )
            for row in rows
        ]
