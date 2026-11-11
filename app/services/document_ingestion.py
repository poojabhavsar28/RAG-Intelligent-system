from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from tenantrag.corpus.simulator import Corpus
from tenantrag.tenancy.pg import PgTenantStore
from tenantrag.text import chunk_by_tokens
from tenantrag.types import Acl, Chunk

MAX_DOCUMENT_CHUNKS = 1000
_LANGUAGE_TAG = re.compile(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*\Z")


@dataclass(frozen=True)
class IngestedDocument:
    document_id: str
    tenant_id: str
    filename: str
    chunk_count: int


class DocumentIngestionService:
    def __init__(self, store: PgTenantStore):
        self._store = store

    def ingest_text(self, tenant_id: str, filename: str, text: str, language: str = "en") -> IngestedDocument:
        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id must not be blank")
        if not _LANGUAGE_TAG.fullmatch(language):
            raise ValueError("language must be a valid two- or three-letter language tag")

        content = text.strip()
        if not content:
            raise ValueError("The uploaded document contains no extractable text")

        parts = chunk_by_tokens(content)
        if len(parts) > MAX_DOCUMENT_CHUNKS:
            raise ValueError(f"Documents may not exceed {MAX_DOCUMENT_CHUNKS} chunks")

        document_id = uuid.uuid4().hex
        acl = Acl(is_public=True)
        corpus = Corpus(
            chunks=[
                Chunk(
                    tenant_id=tenant_id,
                    chunk_id=index,
                    doc_id=document_id,
                    version=1,
                    ordinal=index,
                    text=part,
                    lang=language.lower(),
                    acl=acl,
                    meta={"title": filename},
                )
                for index, part in enumerate(parts)
            ]
        )
        stored = self._store.ingest_corpus(corpus, tenants=[tenant_id])
        if stored != len(parts):
            raise RuntimeError(f"Expected to store {len(parts)} chunks, but stored {stored}")

        return IngestedDocument(document_id, tenant_id, filename, stored)
