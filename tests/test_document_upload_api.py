from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.documents import router
from app.services.document_ingestion import DocumentIngestionService
from app.core.security import hash_token


class FakeStore:
    def __init__(self):
        self.corpus = None
        self.tenants = None

    def ingest_corpus(self, corpus, *, tenants):
        self.corpus = corpus
        self.tenants = list(tenants)
        return len(corpus.chunks)


class FakeTenantRepository:
    def resolve_api_key(self, token_hash):
        return "tenant-a" if token_hash == hash_token("tenant-key") else None


def build_client():
    store = FakeStore()
    app = FastAPI()
    app.include_router(router)
    app.state.database_backend = "postgres"
    app.state.tenant_repo = FakeTenantRepository()
    app.state.document_ingestion_service = DocumentIngestionService(store)
    return TestClient(app), store


def test_upload_requires_tenant_api_key():
    client, _ = build_client()

    response = client.post(
        "/documents/upload",
        files={"file": ("policy.txt", b"Refunds are accepted within 30 days.", "text/plain")},
    )

    assert response.status_code == 401


def test_upload_persists_tenant_scoped_public_document_chunks():
    client, store = build_client()

    response = client.post(
        "/documents/upload",
        headers={"X-Tenant-API-Key": "tenant-key"},
        files={"file": ("policy.txt", b"Refunds are accepted within 30 days.", "text/plain")},
        data={"language": "en"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["tenant_id"] == "tenant-a"
    assert body["filename"] == "policy.txt"
    assert body["chunks"] == 1
    assert store.tenants == ["tenant-a"]
    assert len(store.corpus.chunks) == 1
    chunk = store.corpus.chunks[0]
    assert chunk.tenant_id == "tenant-a"
    assert chunk.doc_id == body["document_id"]
    assert chunk.acl.is_public
    assert chunk.lang == "en"
    assert chunk.meta == {"title": "policy.txt"}


def test_upload_rejects_unsupported_file_types():
    client, store = build_client()

    response = client.post(
        "/documents/upload",
        headers={"X-Tenant-API-Key": "tenant-key"},
        files={"file": ("policy.docx", b"not supported", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert store.corpus is None


def test_upload_rejects_invalid_tenant_key():
    client, store = build_client()

    response = client.post(
        "/documents/upload",
        headers={"X-Tenant-API-Key": "invalid"},
        files={"file": ("policy.txt", b"Refunds are accepted within 30 days.", "text/plain")},
    )

    assert response.status_code == 401
    assert store.corpus is None


def test_upload_rejects_unreadable_pdf():
    client, store = build_client()

    response = client.post(
        "/documents/upload",
        headers={"X-Tenant-API-Key": "tenant-key"},
        files={"file": ("policy.pdf", b"not a PDF", "application/pdf")},
    )

    assert response.status_code == 400
    assert store.corpus is None
