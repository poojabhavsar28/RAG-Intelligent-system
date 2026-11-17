"""PostgreSQL/pgvector backend: ingestion, hybrid (dense + tsvector) retrieval, ACL admin.

Every statement runs inside ``tenant_transaction`` so row-level security is
active; queries ALSO carry ``tenant_id`` explicitly (partition pruning + defense
in depth). Principal ids are mapped to the text ids used by ``document_acl``:
users -> ``str(user_id)``, groups -> ``str(group_id)``, and every authenticated
tenant user holds the ``tenant_member`` role (used for public documents).
"""
from __future__ import annotations

import hashlib
import time
import uuid
from typing import Iterable, Sequence

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import Engine, bindparam, text

from app.infrastructure.postgres.tenant_db import tenant_transaction
from tenantrag.corpus.simulator import Corpus
from tenantrag.retrieval.fusion import reciprocal_rank_fusion
from tenantrag.retrieval.hybrid import RetrievalResult
from tenantrag.text import Embedder, chunk_by_tokens, tokenize
from tenantrag.types import Acl, Chunk, Hit, Principal

_NS = uuid.UUID("6f1d5c0e-7a54-4e0c-9d2e-3b1d9a5f0c11")
PUBLIC_ROLE = "tenant_member"


def doc_uuid(tenant_id: str, doc_id: str) -> uuid.UUID:
    return uuid.uuid5(_NS, f"{tenant_id}/{doc_id}")


def _acl_rows(tenant_id: str, doc: uuid.UUID, acl: Acl) -> list[dict]:
    rows = []
    if acl.is_public:
        rows.append(("role", PUBLIC_ROLE, "read"))
    rows += [("group", str(g), "read") for g in acl.allow_groups]
    rows += [("user", str(u), "read") for u in acl.allow_users]
    rows += [("group", str(g), "deny") for g in acl.deny_groups]
    return [{"t": tenant_id, "d": doc, "pt": a, "pi": b, "perm": c} for a, b, c in rows]


_ACL_SQL = """
  AND d.status = 'active' AND d.deleted_at IS NULL
  AND d.trust_tier >= 1 AND d.injection_state = 'clean'
  AND EXISTS (
        SELECT 1 FROM document_acl a
        WHERE a.tenant_id = c.tenant_id AND a.document_id = c.document_id
          AND a.permission IN ('read', 'write', 'admin')
          AND ((a.principal_type = 'user'  AND a.principal_id = :uid)
            OR (a.principal_type = 'group' AND a.principal_id = ANY(CAST(:gids AS TEXT[])))
            OR (a.principal_type = 'role'  AND a.principal_id = ANY(CAST(:roles AS TEXT[])))))
  AND NOT EXISTS (
        SELECT 1 FROM document_acl x
        WHERE x.tenant_id = c.tenant_id AND x.document_id = c.document_id AND x.permission = 'deny'
          AND ((x.principal_type = 'user'  AND x.principal_id = :uid)
            OR (x.principal_type = 'group' AND x.principal_id = ANY(CAST(:gids AS TEXT[])))
            OR (x.principal_type = 'role'  AND x.principal_id = ANY(CAST(:roles AS TEXT[])))))
"""

_DENSE = text(f"""
  SELECT c.chunk_id::text AS chunk_id, COALESCE(d.external_id, c.document_id::text) AS doc_id,
         c.ordinal, c.content, 1 - (c.embedding <=> :emb) AS score
  FROM document_chunks c JOIN documents d ON d.tenant_id = c.tenant_id AND d.document_id = c.document_id
  WHERE c.tenant_id = :tid AND c.embedding IS NOT NULL {_ACL_SQL}
  ORDER BY c.embedding <=> :emb LIMIT :n
""").bindparams(bindparam("emb", type_=Vector(768)))

_LEXICAL = text(f"""
  SELECT c.chunk_id::text AS chunk_id, COALESCE(d.external_id, c.document_id::text) AS doc_id,
         c.ordinal, c.content, ts_rank_cd(c.tsv, q.query) AS score
  FROM document_chunks c JOIN documents d ON d.tenant_id = c.tenant_id AND d.document_id = c.document_id,
       to_tsquery('simple', :q) AS q(query)
  WHERE c.tenant_id = :tid AND c.tsv @@ q.query {_ACL_SQL}
  ORDER BY score DESC LIMIT :n
""")


def _or_query(query: str) -> str | None:
    toks = sorted({t for t in tokenize(query) if t})
    return " | ".join(f"'{t}'" for t in toks) if toks else None  # tokens contain only letters/marks/digits


class PgTenantStore:
    """Ingestion and ACL administration."""

    def __init__(self, engine: Engine, embedder: Embedder):
        if embedder.dim != 768:
            raise ValueError("document_chunks.embedding is vector(768); use a 768-d embedder")
        self.engine, self.embedder = engine, embedder

    def ingest_corpus(self, corpus: Corpus, *, batch: int = 500, tenants: Iterable[str] | None = None) -> int:
        want = set(tenants) if tenants is not None else None
        by_t: dict[str, list[Chunk]] = {}
        for c in corpus.chunks:
            if want is None or c.tenant_id in want:
                by_t.setdefault(c.tenant_id, []).append(c)
        total = 0
        ins_chunk = text(
            "INSERT INTO document_chunks (tenant_id, chunk_id, document_id, ordinal, content, content_hash, "
            "language, embedding, token_count) VALUES (:t, :cid, :d, :o, :txt, :h, 'en', :emb, :n)"
        ).bindparams(bindparam("emb", type_=Vector(768)))
        for t, chunks in by_t.items():
            with tenant_transaction(self.engine, t) as conn:
                conn.execute(
                    text("INSERT INTO tenants (tenant_id, name) VALUES (:t, :t) ON CONFLICT DO NOTHING"), {"t": t}
                )
                docs = {c.doc_id for c in chunks}
                conn.execute(
                    text(
                        "INSERT INTO documents (tenant_id, document_id, external_id, title, checksum, language) "
                        "VALUES (:t, :d, :x, :x, :h, 'en') ON CONFLICT DO NOTHING"
                    ),
                    [{"t": t, "d": doc_uuid(t, x), "x": x, "h": hashlib.sha256(f"{t}/{x}".encode()).hexdigest()} for x in docs],
                )
                seen: set[str] = set()
                acl_rows: list[dict] = []
                for c in chunks:
                    if c.doc_id not in seen:
                        seen.add(c.doc_id)
                        acl_rows += _acl_rows(t, doc_uuid(t, c.doc_id), c.acl)
                if acl_rows:
                    conn.execute(
                        text(
                            "INSERT INTO document_acl (tenant_id, document_id, principal_type, principal_id, permission) "
                            "VALUES (:t, :d, :pt, :pi, :perm) ON CONFLICT DO NOTHING"
                        ),
                        acl_rows,
                    )
                for i in range(0, len(chunks), batch):
                    part = chunks[i : i + batch]
                    vecs = self.embedder.embed([c.text for c in part])
                    conn.execute(
                        ins_chunk,
                        [
                            {
                                "t": t, "cid": uuid.uuid5(_NS, f"{t}/chunk/{c.chunk_id}"), "d": doc_uuid(t, c.doc_id),
                                "o": c.ordinal, "txt": c.text, "h": hashlib.sha256(c.text.encode()).hexdigest(),
                                "emb": v.tolist(), "n": len(tokenize(c.text)),
                            }
                            for c, v in zip(part, vecs)
                        ],
                    )
                    total += len(part)
        return total

    def ingest_document(
        self,
        tenant_id: str,
        external_id: str,
        title: str,
        content: str,
        *,
        acl: Acl | None = None,
        source_uri: str | None = None,
        language: str = "en",
        max_chunk_tokens: int = 400,
        overlap: int = 60,
        trust_tier: int = 1,
    ) -> dict:
        """Atomically upsert one tenant document and its embedded chunks.

        Embeddings are computed before opening the database transaction. The document is
        visible only after every chunk and ACL row has been committed successfully.
        """
        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id is required")
        if not external_id or not external_id.strip():
            raise ValueError("external_id is required")
        if not title or not title.strip():
            raise ValueError("title is required")
        if not content or not content.strip():
            raise ValueError("content is required")
        if trust_tier not in (1, 2, 3):
            raise ValueError("trust_tier must be 1, 2 or 3")

        parts = chunk_by_tokens(content, max_tokens=max_chunk_tokens, overlap=overlap)
        if not parts:
            raise ValueError("document has no indexable content")
        vectors = self.embedder.embed(parts)
        if vectors.shape != (len(parts), 768) or not np.isfinite(vectors).all():
            raise ValueError("embedder returned invalid vectors for pgvector(768)")

        doc = doc_uuid(tenant_id, external_id)
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        policy = acl or Acl(is_public=True)
        suspect = any(__import__("tenantrag.security.guards", fromlist=["is_suspect"]).is_suspect(p) for p in parts)
        chunk_rows = []
        for ordinal, (part, vector) in enumerate(zip(parts, vectors)):
            chunk_rows.append({
                "t": tenant_id,
                "cid": uuid.uuid5(_NS, f"{tenant_id}/chunk/{external_id}/{ordinal}"),
                "d": doc,
                "o": ordinal,
                "txt": part,
                "h": hashlib.sha256(part.encode("utf-8")).hexdigest(),
                "lang": language,
                "emb": vector.tolist(),
                "n": len(tokenize(part)),
            })

        insert_chunk = text(
            "INSERT INTO document_chunks (tenant_id, chunk_id, document_id, ordinal, content, "
            "content_hash, language, embedding, token_count) "
            "VALUES (:t, :cid, :d, :o, :txt, :h, :lang, :emb, :n)"
        ).bindparams(bindparam("emb", type_=Vector(768)))

        with tenant_transaction(self.engine, tenant_id) as conn:
            conn.execute(
                text(
                    "INSERT INTO documents (tenant_id, document_id, external_id, title, source_uri, "
                    "checksum, language, status, trust_tier, injection_state) "
                    "VALUES (:t, :d, :x, :title, :uri, :checksum, :lang, 'pending', :trust, :injection) "
                    "ON CONFLICT (tenant_id, document_id) DO UPDATE SET "
                    "external_id=EXCLUDED.external_id, title=EXCLUDED.title, source_uri=EXCLUDED.source_uri, "
                    "checksum=EXCLUDED.checksum, language=EXCLUDED.language, status='pending', "
                    "deleted_at=NULL, trust_tier=EXCLUDED.trust_tier, injection_state=EXCLUDED.injection_state, "
                    "acl_version=documents.acl_version+1, updated_at=now()"
                ),
                {
                    "t": tenant_id, "d": doc, "x": external_id, "title": title, "uri": source_uri,
                    "checksum": checksum, "lang": language, "trust": trust_tier,
                    "injection": "suspect" if suspect else "clean",
                },
            )
            conn.execute(
                text("DELETE FROM document_chunks WHERE tenant_id=:t AND document_id=:d"),
                {"t": tenant_id, "d": doc},
            )
            conn.execute(
                text("DELETE FROM document_acl WHERE tenant_id=:t AND document_id=:d"),
                {"t": tenant_id, "d": doc},
            )
            acl_rows = _acl_rows(tenant_id, doc, policy)
            if acl_rows:
                conn.execute(
                    text(
                        "INSERT INTO document_acl (tenant_id, document_id, principal_type, principal_id, permission) "
                        "VALUES (:t, :d, :pt, :pi, :perm)"
                    ),
                    acl_rows,
                )
            conn.execute(insert_chunk, chunk_rows)
            conn.execute(
                text(
                    "UPDATE documents SET status='active', updated_at=now() "
                    "WHERE tenant_id=:t AND document_id=:d"
                ),
                {"t": tenant_id, "d": doc},
            )

        return {
            "document_id": external_id,
            "checksum": checksum,
            "chunks": len(chunk_rows),
            "status": "active",
            "injection_suspect": suspect,
        }

    def set_acl(self, tenant_id: str, doc_id: str, acl: Acl) -> int:
        """Replace a document's ACL atomically, bump acl_version, emit an outbox event.

        Revocation is enforced on the very next query (retrieval reads the authoritative
        ``document_acl``); the event lets caches and workers invalidate eagerly.
        """
        d = doc_uuid(tenant_id, doc_id)
        with tenant_transaction(self.engine, tenant_id) as conn:
            conn.execute(text("DELETE FROM document_acl WHERE tenant_id=:t AND document_id=:d"), {"t": tenant_id, "d": d})
            rows = _acl_rows(tenant_id, d, acl)
            if rows:
                conn.execute(
                    text(
                        "INSERT INTO document_acl (tenant_id, document_id, principal_type, principal_id, permission) "
                        "VALUES (:t, :d, :pt, :pi, :perm)"
                    ),
                    rows,
                )
            ver = conn.execute(
                text(
                    "UPDATE documents SET acl_version = acl_version + 1, updated_at = now() "
                    "WHERE tenant_id=:t AND document_id=:d RETURNING acl_version"
                ),
                {"t": tenant_id, "d": d},
            ).scalar_one()
            conn.execute(
                text("INSERT INTO acl_events (tenant_id, document_id, acl_version) VALUES (:t, :d, :v)"),
                {"t": tenant_id, "d": d, "v": ver},
            )
        return int(ver)

    def soft_delete(self, tenant_id: str, doc_id: str) -> None:
        with tenant_transaction(self.engine, tenant_id) as conn:
            conn.execute(
                text(
                    "UPDATE documents SET status='deleted', deleted_at=now() WHERE tenant_id=:t AND document_id=:d "
                    "AND NOT legal_hold"
                ),
                {"t": tenant_id, "d": doc_uuid(tenant_id, doc_id)},
            )

    def pending_acl_events(self, tenant_id: str) -> list[tuple[str, int]]:
        with tenant_transaction(self.engine, tenant_id) as conn:
            rows = conn.execute(
                text("SELECT document_id::text, acl_version FROM acl_events WHERE tenant_id=:t AND applied_at IS NULL"),
                {"t": tenant_id},
            ).all()
        return [(r[0], r[1]) for r in rows]


class PgHybridRetriever:
    """Dense (HNSW cosine) + lexical (GIN tsvector) -> RRF, ACL enforced in SQL under RLS."""

    def __init__(self, engine: Engine, embedder: Embedder, *, candidates: int = 40, rrf_k: int = 60, audit: bool = False):
        self.engine, self.embedder = engine, embedder
        self.candidates, self.rrf_k, self.audit = candidates, rrf_k, audit

    @staticmethod
    def _params(p: Principal) -> dict:
        return {"tid": p.tenant_id, "uid": str(p.user_id), "gids": [str(g) for g in p.group_ids], "roles": [PUBLIC_ROLE]}

    @staticmethod
    def _hit(row, tenant_id: str, source: str) -> Hit:
        return Hit(Chunk(tenant_id, row["chunk_id"], row["doc_id"], 1, row["ordinal"], row["content"]), float(row["score"]), source)

    def retrieve(
        self,
        p: Principal,
        query: str,
        k: int = 10,
        *,
        dense: bool = True,
        lexical: bool = True,
        ef_search: int | None = None,
        exact: bool = False,
    ) -> RetrievalResult:
        res = RetrievalResult([])
        t0 = time.perf_counter()
        legs: list[list[Hit]] = []
        params = self._params(p) | {"n": self.candidates}
        with tenant_transaction(self.engine, p.tenant_id) as conn:
            if dense:
                if ef_search:
                    conn.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
                if exact:  # ground-truth control: force a sequential scan instead of the ANN index
                    conn.execute(text("SET LOCAL enable_indexscan = off"))
                    conn.execute(text("SET LOCAL enable_bitmapscan = off"))
                emb = self.embedder.embed_query(query).tolist()
                rows = conn.execute(_DENSE, params | {"emb": emb}).mappings().all()
                legs.append([self._hit(r, p.tenant_id, "dense") for r in rows])
            q = _or_query(query)
            if lexical and q:
                rows = conn.execute(_LEXICAL, params | {"q": q}).mappings().all()
                legs.append([self._hit(r, p.tenant_id, "lexical") for r in rows])
            if self.audit and legs:
                top = [h.chunk.doc_id for h in (legs[0][:k])]
                conn.execute(
                    text("INSERT INTO audit_log (tenant_id, user_id, query_hash, document_ids) VALUES (:t, :u, :h, :d)"),
                    {"t": p.tenant_id, "u": str(p.user_id), "h": hashlib.sha256(query.encode()).hexdigest()[:16], "d": top},
                )
        fused = reciprocal_rank_fusion(legs, self.rrf_k) if len(legs) > 1 else (legs[0] if legs else [])
        res.hits = fused[:k]
        res.timings_ms = {"total": (time.perf_counter() - t0) * 1e3}
        return res
