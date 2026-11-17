import json

import httpx
import numpy as np
import pytest

from tenantrag.pipeline.llm import OllamaLLM, build_llm
from tenantrag.text import OllamaEmbedder, build_embedder


def test_ollama_llm_posts_chat_and_returns_text():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "Local answer"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OllamaLLM(model="llama3.2:3b", client=client)
    assert llm.complete("system rules", "question") == "Local answer"
    assert seen["url"].endswith("/api/chat")
    assert seen["payload"]["model"] == "llama3.2:3b"
    assert seen["payload"]["stream"] is False
    assert seen["payload"]["messages"][0]["role"] == "system"
    client.close()


def test_ollama_llm_rejects_empty_response():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"message": {"content": ""}}))
    )
    with pytest.raises(RuntimeError, match="empty or invalid"):
        OllamaLLM(client=client).complete("s", "p")
    client.close()


def test_ollama_embedder_batches_and_normalizes():
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload["input"])
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0] for _ in payload["input"]]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    emb = OllamaEmbedder(dim=2, batch=2, client=client)
    matrix = emb.embed(["a", "b", "c"])
    assert matrix.shape == (3, 2)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    assert [len(x) for x in calls] == [2, 1]
    assert np.allclose(emb.embed_query("q"), [0.6, 0.8])
    client.close()


def test_ollama_embedder_rejects_wrong_dimensions():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"embeddings": [[1.0, 2.0]]}))
    )
    with pytest.raises(ValueError, match="dimension mismatch"):
        OllamaEmbedder(dim=3, client=client).embed(["text"])
    client.close()


def test_provider_factories_are_explicit_and_ollama_first(monkeypatch):
    monkeypatch.delenv("TENANTRAG_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("TENANTRAG_EMBEDDING_PROVIDER", raising=False)
    assert isinstance(build_llm("extractive"), object)
    assert build_llm("extractive").name == "extractive-stub"
    assert isinstance(build_embedder("hashing"), object)
    with pytest.raises(ValueError, match="Unsupported TENANTRAG_LLM_PROVIDER"):
        build_llm("gemini")
    with pytest.raises(ValueError, match="Unsupported TENANTRAG_EMBEDDING_PROVIDER"):
        build_embedder("vertex")
