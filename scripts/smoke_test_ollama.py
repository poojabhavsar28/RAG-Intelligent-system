"""Live smoke test for the configured local Ollama models.

Run after `ollama serve` and pulling the configured LLM + embedding models.
This performs real HTTP inference; it is not a mocked unit test.
"""
from __future__ import annotations

import os

import httpx
import numpy as np

from tenantrag.pipeline.llm import OllamaLLM
from tenantrag.text import OllamaEmbedder


def main() -> None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    llm_model = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b")
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    dim = int(os.getenv("OLLAMA_EMBEDDING_DIM", "768"))

    with httpx.Client(timeout=5.0) as client:
        response = client.get(f"{base_url}/api/tags")
        response.raise_for_status()
        available = {item.get("name") for item in response.json().get("models", [])}

    for model in (llm_model, embedding_model):
        if not any(name == model or name.startswith(model + ":") for name in available):
            raise SystemExit(f"Missing Ollama model: {model}. Pull it before running this check.")

    embedder = OllamaEmbedder(model=embedding_model, dim=dim, base_url=base_url)
    llm = OllamaLLM(model=llm_model, base_url=base_url)
    try:
        documents = embedder.embed(["The RAG service stores tenant-scoped documents."])
        query = embedder.embed_query("What does the RAG service store?")
        if documents.shape != (1, dim) or query.shape != (dim,):
            raise RuntimeError("Unexpected embedding output shape")
        if not np.isfinite(documents).all() or not np.isfinite(query).all():
            raise RuntimeError("Embedding output contains non-finite values")
        if not np.isclose(np.linalg.norm(query), 1.0, atol=1e-4):
            raise RuntimeError("Query embedding was not normalized")

        answer = llm.complete(
            "Follow the user's instruction exactly.",
            "Reply with exactly this single word: READY",
        )
        if "READY" not in answer.upper():
            raise RuntimeError(f"Unexpected Ollama chat response: {answer!r}")
        print(f"PASS: Ollama chat model {llm_model}")
        print(f"PASS: Ollama embedding model {embedding_model} ({dim} dimensions)")
    finally:
        embedder.close()
        llm.close()


if __name__ == "__main__":
    main()
