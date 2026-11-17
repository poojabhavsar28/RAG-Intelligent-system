"""LLM interface. ``ExtractiveLLM`` is a deterministic, offline stand-in used by tests and
benchmarks (it quotes the best-matching sentence and cites it), so pipeline logic can be
measured without model variance. ``GeminiLLM`` is the production adapter (Vertex AI)."""
from __future__ import annotations

import re
from typing import Protocol

from tenantrag.text import tokenize


class LLM(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> str: ...


_DOC = re.compile(r'<document id="([^"]+)">\n(.*?)\n</document>', re.S)
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_STOP = {"what", "is", "the", "for", "of", "tell", "me", "how", "long", "under", "a", "an", "to", "in", "and"}


class ExtractiveLLM:
    name = "extractive-stub"

    def __init__(self, min_overlap: float = 0.34):
        self.min_overlap = min_overlap

    def complete(self, system: str, prompt: str) -> str:
        q = prompt.rsplit("QUESTION:", 1)[-1].strip()
        qt = {t for t in tokenize(q) if t not in _STOP}
        best, best_s, best_label = "", 0.0, ""
        for label, body in _DOC.findall(prompt):
            for sent in _SENT.split(body):
                st = set(tokenize(sent))
                if not st or not qt:
                    continue
                s = len(qt & st) / len(qt)
                if s > best_s:
                    best, best_s, best_label = sent.strip(), s, label
        if best_s < self.min_overlap:
            return "I don't know based on the provided documents."
        return f"{best} [{best_label}]"


class OllamaLLM:
    """Local Ollama chat completion adapter; no cloud credentials or paid API required."""

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        client=None,
    ):
        import httpx

        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def complete(self, system: str, prompt: str) -> str:
        response = self._client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload.get("message", {}).get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Ollama returned an empty or invalid chat response")
        return answer.strip()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def build_llm(provider: str | None = None) -> LLM:
    """Build the configured provider. Ollama is the default; extractive is test-only."""
    import os

    selected = (provider or os.getenv("TENANTRAG_LLM_PROVIDER", "ollama")).strip().lower()
    if selected == "ollama":
        return OllamaLLM(
            model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            timeout=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
        )
    if selected == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"Unsupported TENANTRAG_LLM_PROVIDER: {selected!r}; use 'ollama' or 'extractive'")
