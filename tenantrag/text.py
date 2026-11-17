"""Unicode-aware tokenizer and a deterministic hashing embedder.

Python's ``\\w`` splits Hindi/Gujarati words at combining vowel signs (category
Mc/Mn), so tokenization walks characters and keeps letters, marks and digits.

``HashingEmbedder`` is a dependency-free baseline (character n-grams hashed
into a fixed-size signed vector). It is lexically fuzzy, NOT semantic and NOT
cross-lingual. It exists so the isolation, ACL, cache and benchmark code can be
tested deterministically; production uses ``OllamaEmbedder``.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable, Protocol, Sequence

import numpy as np


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for ch in text.lower():
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "M", "N"):
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


_WS = re.compile(r"\s+")


def normalize_query(q: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", q)).strip().lower()


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 128, ngrams: tuple[int, ...] = (3, 4)):
        self.dim = dim
        self.ngrams = ngrams

    def _bucket(self, feat: str) -> tuple[int, float]:
        h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        v = int.from_bytes(h, "little")
        return v % self.dim, 1.0 if (v >> 40) & 1 else -1.0

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            padded = f"^{tok}$"
            i, s = self._bucket("w:" + tok)
            vec[i] += 2.0 * s
            for n in self.ngrams:
                for j in range(max(1, len(padded) - n + 1)):
                    i, s = self._bucket(padded[j : j + n])
                    vec[i] += s
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._one(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._one(text)


class OllamaEmbedder:
    """Direct Ollama /api/embed adapter; defaults to nomic-embed-text (768 dimensions)."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        dim: int = 768,
        batch: int = 64,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        client=None,
    ):
        import httpx

        if batch < 1:
            raise ValueError("batch must be >= 1")
        self.model = model
        self.dim = dim
        self._batch = batch
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            response = self._client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": list(texts[i : i + self._batch])},
            )
            response.raise_for_status()
            payload = response.json()
            batch_rows = payload.get("embeddings")
            if not isinstance(batch_rows, list) or len(batch_rows) != len(texts[i : i + self._batch]):
                raise RuntimeError("Ollama returned an invalid embedding batch")
            rows.extend(batch_rows)
        arr = np.asarray(rows, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            actual = arr.shape[1] if arr.ndim == 2 else "invalid"
            raise ValueError(f"Ollama embedding dimension mismatch: expected {self.dim}, received {actual}")
        if not np.isfinite(arr).all():
            raise ValueError("Ollama returned non-finite embedding values")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def build_embedder(provider: str | None = None) -> Embedder:
    """Build the configured embedder. Ollama is default; hashing is deterministic test-only."""
    import os

    selected = (provider or os.getenv("TENANTRAG_EMBEDDING_PROVIDER", "ollama")).strip().lower()
    if selected == "ollama":
        return OllamaEmbedder(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            dim=int(os.getenv("OLLAMA_EMBEDDING_DIM", "768")),
            batch=int(os.getenv("OLLAMA_EMBEDDING_BATCH", "64")),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            timeout=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
        )
    if selected == "hashing":
        return HashingEmbedder(int(os.getenv("TENANTRAG_HASHING_DIM", "128")))
    raise ValueError(
        f"Unsupported TENANTRAG_EMBEDDING_PROVIDER: {selected!r}; use 'ollama' or 'hashing'"
    )
