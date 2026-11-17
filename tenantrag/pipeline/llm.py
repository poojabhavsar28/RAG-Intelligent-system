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


class GeminiLLM:
    """Gemini via Vertex AI. Requires ``google-cloud-aiplatform`` and ADC credentials."""

    def __init__(self, model: str = "gemini-2.0-flash", project: str | None = None, location: str = "us-central1"):
        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=project, location=location)
        self._m = GenerativeModel(model)
        self.name = model

    def complete(self, system: str, prompt: str) -> str:
        from vertexai.generative_models import GenerationConfig

        r = self._m.generate_content(f"{system}\n\n{prompt}", generation_config=GenerationConfig(temperature=0.0))
        return r.text
