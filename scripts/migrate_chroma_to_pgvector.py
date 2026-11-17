"""Import legacy tenant/language Chroma collections into PostgreSQL pgvector.

Requires MySQL chroma_metadata rows and the original Chroma persistence
directories to remain available. Existing 768-dimensional embeddings are
copied directly; no re-embedding or source deletion is performed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import mysql.connector
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, text, bindparam


DIMENSIONS = 768


def _stable_uuid(value: str):
    return uuid5(NAMESPACE_URL, value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _vector(values: Any, *, tenant_id: str, collection_id: str) -> list[float]:
    vector = [float(value) for value in values]
    if len(vector) != DIMENSIONS:
        raise ValueError(
            f"Chroma record {collection_id!r} for tenant {tenant_id!r} has "
            f"{len(vector)} dimensions; PostgreSQL schema requires {DIMENSIONS}"
        )
    return vector


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} must be configured")
    return value


def migrate() -> dict[str, int]:
    mysql_conn = mysql.connector.connect(
        host=_required_env("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=_required_env("MYSQL_USER"),
        password=_required_env("MYSQL_PASSWORD"),
        database=_required_env("MYSQL_DB"),
    )
    pg_url = _required_env("POSTGRES_ADMIN_DATABASE_URL").replace(
        "postgresql://", "postgresql+psycopg://", 1
    )
    pg_engine = create_engine(pg_url, pool_pre_ping=True)
    base_dir = Path(os.getenv("CHROMA_DB_BASE_DIR", "./chroma_db_base")).resolve()
    counts = {"collections": 0, "documents": 0, "chunks": 0, "acl_rows": 0}

    try:
        with pg_engine.begin() as pg:
            can_bypass = pg.execute(text("""
                SELECT r.rolsuper OR r.rolbypassrls
                FROM pg_roles AS r WHERE r.rolname = current_user
            """)).scalar_one()
            if not can_bypass:
                raise RuntimeError(
                    "POSTGRES_ADMIN_DATABASE_URL must use a migration role with BYPASSRLS"
                )

        # Import lazily so unit tests for pure helpers do not need Chroma's
        # native runtime dependencies installed.
        import chromadb

        source = mysql_conn.cursor(dictionary=True)
        try:
            source.execute(
                "SELECT tenant_id, lang, db_path FROM chroma_metadata ORDER BY tenant_id, lang"
            )
            collections = source.fetchall()
        finally:
            source.close()

        for record in collections:
            tenant_id = record["tenant_id"]
            if not isinstance(tenant_id, str) or not tenant_id.strip():
                raise ValueError(f"Invalid tenant ID in chroma_metadata: {tenant_id!r}")
            language = record["lang"]
            if not language:
                raise ValueError(f"Missing language for tenant {tenant_id!r}")

            collection_path = Path(record["db_path"]).resolve()
            if os.path.commonpath([str(base_dir), str(collection_path)]) != str(base_dir):
                raise ValueError(
                    f"Chroma path {collection_path} is outside CHROMA_DB_BASE_DIR {base_dir}"
                )
            if not collection_path.is_dir():
                raise FileNotFoundError(
                    f"Chroma collection path does not exist: {collection_path}"
                )

            client = chromadb.PersistentClient(path=str(collection_path))
            collection = client.get_collection(name="documents")
            data = collection.get(include=["documents", "metadatas", "embeddings"])
            ids = data.get("ids") or []
            texts = data.get("documents") or []
            metadatas = data.get("metadatas") or []
            embeddings = data.get("embeddings")
            if embeddings is None and ids:
                raise ValueError(f"Chroma collection has no embeddings: {collection_path}")
            if not ids:
                counts["collections"] += 1
                continue

            grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for index, legacy_id in enumerate(ids):
                metadata = metadatas[index] or {}
                metadata_tenant = metadata.get("tenant_id")
                if metadata_tenant is not None and str(metadata_tenant) != tenant_id:
                    raise ValueError(
                        f"Tenant metadata mismatch in {legacy_id!r}: "
                        f"{metadata_tenant!r} != {tenant_id!r}"
                    )
                source_name = str(metadata.get("source") or "legacy-chroma")
                source_type = str(metadata.get("type") or metadata.get("file_type") or "document")
                grouped.setdefault((source_name, source_type), []).append({
                    "legacy_id": str(legacy_id),
                    "content": texts[index] or "",
                    "metadata": _json_safe(metadata),
                    "embedding": _vector(
                        embeddings[index], tenant_id=tenant_id, collection_id=str(legacy_id)
                    ),
                    "chunk_index": int(metadata.get("chunk_index", index)),
                })

            for (source_name, source_type), chunks in grouped.items():
                chunks.sort(key=lambda item: (item["chunk_index"], item["legacy_id"]))
                document_seed = f"{tenant_id}|{language}|{source_type}|{source_name}"
                document_id = _stable_uuid("legacy-chroma-document|" + document_seed)
                chunk_payload = []
                content_hashes = []
                for ordinal, chunk in enumerate(chunks):
                    content_hash = hashlib.sha256(chunk["content"].encode("utf-8")).hexdigest()
                    content_hashes.append(content_hash)
                    chunk_id = _stable_uuid(
                        f"legacy-chroma-chunk|{tenant_id}|{language}|{chunk['legacy_id']}"
                    )
                    chunk_metadata = dict(chunk["metadata"])
                    chunk_metadata["legacy_chroma_id"] = chunk["legacy_id"]
                    chunk_payload.append({
                        "tenant_id": tenant_id,
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "ordinal": ordinal,
                        "content": chunk["content"],
                        "content_hash": content_hash,
                        "language": language,
                        "embedding": chunk["embedding"],
                        "token_count": None,
                        "metadata": json.dumps(chunk_metadata, ensure_ascii=False),
                    })

                checksum = hashlib.sha256(
                    (document_seed + "|" + "|".join(content_hashes)).encode("utf-8")
                ).hexdigest()
                document_metadata = json.dumps({
                    "legacy_source": source_name,
                    "legacy_type": source_type,
                    "legacy_vector_store": "chroma",
                })

                chunk_statement = text("""
                    INSERT INTO document_chunks
                        (tenant_id, chunk_id, document_id, ordinal, content,
                         content_hash, language, embedding, token_count, metadata)
                    VALUES
                        (:tenant_id, :chunk_id, :document_id, :ordinal, :content,
                         :content_hash, :language, :embedding, :token_count,
                         CAST(:metadata AS JSONB))
                    ON CONFLICT (tenant_id, chunk_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        content_hash = EXCLUDED.content_hash,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                """).bindparams(bindparam("embedding", type_=Vector(DIMENSIONS)))

                with pg_engine.begin() as pg:
                    pg.execute(text("""
                        INSERT INTO documents
                            (tenant_id, document_id, external_id, title, language,
                             checksum, status, metadata)
                        VALUES
                            (:tenant_id, :document_id, :external_id, :title, :language,
                             :checksum, 'active', CAST(:metadata AS JSONB))
                        ON CONFLICT (tenant_id, document_id) DO UPDATE SET
                            title = EXCLUDED.title,
                            checksum = EXCLUDED.checksum,
                            status = 'active',
                            metadata = EXCLUDED.metadata,
                            updated_at = now()
                    """), {
                        "tenant_id": tenant_id,
                        "document_id": document_id,
                        "external_id": source_name,
                        "title": source_name,
                        "language": language,
                        "checksum": checksum,
                        "metadata": document_metadata,
                    })
                    pg.execute(chunk_statement, chunk_payload)
                    # Legacy Chroma was isolated by tenant but had no per-user
                    # ACL. This tenant-local role preserves that access model.
                    pg.execute(text("""
                        INSERT INTO document_acl
                            (tenant_id, document_id, principal_type, principal_id, permission)
                        VALUES (:tenant_id, :document_id, 'role', 'tenant_member', 'read')
                        ON CONFLICT DO NOTHING
                    """), {"tenant_id": tenant_id, "document_id": document_id})

                counts["documents"] += 1
                counts["chunks"] += len(chunk_payload)
                counts["acl_rows"] += 1

            counts["collections"] += 1

        return counts
    finally:
        mysql_conn.close()
        pg_engine.dispose()


if __name__ == "__main__":
    result = migrate()
    print("Chroma to pgvector migration counts:")
    for table, count in result.items():
        print(f"  {table}: {count}")
