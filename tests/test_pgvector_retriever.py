import pytest

from app.domain.retrieval import RetrievalContext
from app.infrastructure.vectorstore.pgvector_store import PgVectorRetriever
from tenantrag.text import OllamaEmbedder


class StaticEmbedder:
    def __init__(self, vector):
        self.vector = vector
        self.calls = 0

    def embed_query(self, text):
        self.calls += 1
        return self.vector


def test_pgvector_retriever_rejects_non_768_dimensional_embeddings_before_db_access():
    embedder = StaticEmbedder([0.0] * 767)
    retriever = PgVectorRetriever(engine=None, embedder=embedder)

    with pytest.raises(ValueError, match="Expected a 768-dimensional"):
        retriever.retrieve(RetrievalContext(tenant_id="tenant-a"), "hello")

    assert embedder.calls == 1


def test_pgvector_retriever_rejects_invalid_k_before_embedding():
    embedder = StaticEmbedder([0.0] * 768)
    retriever = PgVectorRetriever(engine=None, embedder=embedder)

    with pytest.raises(ValueError, match="k must be between 1 and 100"):
        retriever.retrieve(RetrievalContext(tenant_id="tenant-a"), "hello", k=101)

    assert embedder.calls == 0


def test_pgvector_retriever_returns_empty_for_blank_query_without_embedding():
    embedder = StaticEmbedder([0.0] * 768)
    retriever = PgVectorRetriever(engine=None, embedder=embedder)

    assert retriever.retrieve(RetrievalContext(tenant_id="tenant-a"), "   ") == []
    assert embedder.calls == 0


def test_pgvector_retriever_requires_schema_dimension():
    with pytest.raises(ValueError, match="requires 768-dimensional"):
        PgVectorRetriever(engine=None, embedder=StaticEmbedder([]), dimensions=1024)


def test_ollama_embedder_uses_configured_server_url(monkeypatch):
    import sys
    import types

    created = {}

    class FakeOllamaEmbeddings:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langchain_ollama",
        types.SimpleNamespace(OllamaEmbeddings=FakeOllamaEmbeddings),
    )

    OllamaEmbedder(base_url="http://ollama.test:11434")

    assert created == {
        "model": "nomic-embed-text",
        "base_url": "http://ollama.test:11434",
    }
