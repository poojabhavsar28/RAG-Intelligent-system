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
    """Adapter for the existing Nomic/Ollama embedding setup (768-d)."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        dim: int = 768,
        batch: int = 64,
        base_url: str | None = None,
    ):
        from langchain_ollama import OllamaEmbeddings  # lazy: heavy optional dep

        options = {"model": model}
        if base_url:
            options["base_url"] = base_url
        self._emb = OllamaEmbeddings(**options)
        self.dim = dim
        self._batch = batch

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self._batch):  # embedding batching at ingestion
            rows.extend(self._emb.embed_documents(list(texts[i : i + self._batch])))
        arr = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)

    def embed_query(self, text: str) -> np.ndarray:
        v = np.asarray(self._emb.embed_query(text), dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v


def chunk_by_tokens(text: str, max_tokens: int = 120, overlap: int = 20) -> list[str]:
    """Simple whitespace-token chunker used by ingestion when no structure is known."""
    words = text.split()
    if len(words) <= max_tokens:
        return [text.strip()] if text.strip() else []
    step = max(1, max_tokens - overlap)
    return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), step) if words[i : i + max_tokens]]
