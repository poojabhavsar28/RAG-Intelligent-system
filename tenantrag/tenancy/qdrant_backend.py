"""Qdrant backend: ONE collection, payload-based multitenancy (the comparison system).

Isolation and ACLs are payload filters evaluated inside Qdrant's filter-aware HNSW
traversal (not post-filtering). ``tenant_id`` is indexed with ``is_tenant=True`` so
Qdrant co-locates a tenant's vectors. ACL fields mirror ``Acl``: deny wins, then
public / group / user grants. Works against a server or ``QdrantClient(":memory:")``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from tenantrag.types import Chunk, Hit, Principal


class QdrantTenantStore:
    def __init__(self, client: QdrantClient, dim: int, collection: str = "tenantrag_chunks"):
        self.client, self.dim, self.collection = client, dim, collection
        self._chunks: dict[int, Chunk] = {}  # payload holds ids only; text stays in the doc store
        if not client.collection_exists(collection):
            client.create_collection(
                collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )
        try:
            client.create_payload_index(
                collection, "tenant_id", models.KeywordIndexParams(type="keyword", is_tenant=True)
            )
        except Exception:  # local mode / older servers: keyword index without the tenant hint
            client.create_payload_index(collection, "tenant_id", models.PayloadSchemaType.KEYWORD)

    def ingest(self, chunks: Sequence[Chunk], vectors: np.ndarray, batch: int = 256) -> None:
        for i in range(0, len(chunks), batch):
            pts = []
            for c, v in zip(chunks[i : i + batch], vectors[i : i + batch]):
                self._chunks[c.chunk_id] = c
                pts.append(
                    models.PointStruct(
                        id=c.chunk_id,
                        vector=v.tolist(),
                        payload={
                            "tenant_id": c.tenant_id,
                            "doc_id": c.doc_id,
                            "is_public": c.acl.is_public,
                            "allow_groups": sorted(c.acl.allow_groups),
                            "allow_users": sorted(c.acl.allow_users),
                            "deny_groups": sorted(c.acl.deny_groups),
                        },
                    )
                )
            self.client.upsert(self.collection, points=pts, wait=True)

    @staticmethod
    def _filter(p: Principal) -> models.Filter:
        any_of = [models.FieldCondition(key="is_public", match=models.MatchValue(value=True))]
        if p.group_ids:
            any_of.append(models.FieldCondition(key="allow_groups", match=models.MatchAny(any=list(p.group_ids))))
        any_of.append(models.FieldCondition(key="allow_users", match=models.MatchAny(any=[p.user_id])))
        must_not = []
        if p.group_ids:
            must_not.append(models.FieldCondition(key="deny_groups", match=models.MatchAny(any=list(p.group_ids))))
        return models.Filter(
            must=[
                models.FieldCondition(key="tenant_id", match=models.MatchValue(value=p.tenant_id)),
                models.Filter(should=any_of),
            ],
            must_not=must_not or None,
        )

    def dense(self, p: Principal, qvec: np.ndarray, k: int) -> list[Hit]:
        res = self.client.query_points(
            self.collection, query=qvec.tolist(), query_filter=self._filter(p), limit=k, with_payload=False
        ).points
        return [Hit(self._chunks[r.id], float(r.score), "dense") for r in res]
