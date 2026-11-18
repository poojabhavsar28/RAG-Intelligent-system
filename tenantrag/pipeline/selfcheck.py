"""Corrective loop: judge whether retrieved context answers the question; if not, repair
the query and retry; if still not, abstain with a reason (Corrective-RAG / Self-RAG style).

The judge is a transparent coverage heuristic (share of the question's content terms that
appear in the context). Swap in an LLM judge by implementing ``judge``; the loop and its
measured cost/benefit (bench/selfcheck_ablation.py) stay the same."""
from __future__ import annotations

import difflib
from dataclasses import dataclass

from tenantrag.text import tokenize
from tenantrag.types import Hit

_STOP = {"what", "is", "the", "for", "of", "tell", "me", "how", "long", "under", "a", "an", "to", "in", "and",
         "are", "do", "does", "who", "when", "which"}


def content_terms(query: str) -> list[str]:
    return [t for t in tokenize(query) if t not in _STOP]


@dataclass
class Judgement:
    sufficient: bool
    coverage: float
    missing: list[str]


def judge(query: str, hits: list[Hit], top: int = 3, threshold: float = 0.99, per_chunk: bool = True) -> Judgement:
    """Coverage of the question's content terms by the context.

    ``per_chunk=True`` (default) requires ONE chunk to cover the terms, so a sibling document that
    happens to contain the same digit cannot make the context look sufficient. Multi-hop questions
    should be judged per decomposed sub-query (or with per_chunk=False)."""
    terms = content_terms(query)
    if not terms or not hits:
        return Judgement(False, 0.0, terms)
    sets = [set(tokenize(h.chunk.text)) for h in hits[:top]]
    if not per_chunk:
        sets = [set().union(*sets)]
    best_missing = min(([t for t in terms if t not in s] for s in sets), key=len)
    cov = 1 - len(best_missing) / len(terms)
    return Judgement(cov >= threshold, cov, best_missing)


def repair_query(query: str, hits: list[Hit], missing: list[str]) -> str | None:
    """Replace unmatched terms with their closest vocabulary item from the retrieved pool
    (fixes typos); returns None when nothing could be repaired."""
    vocab: set[str] = set()
    for h in hits[:10]:
        vocab.update(tokenize(h.chunk.text))
    if not vocab:
        return None
    new, changed = query, False
    for t in missing:
        m = difflib.get_close_matches(t, vocab, n=1, cutoff=0.75)
        if m and m[0] != t:
            new = new.replace(t, m[0])
            changed = True
    return new if changed else None
