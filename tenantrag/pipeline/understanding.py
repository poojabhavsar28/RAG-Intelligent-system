"""Query understanding: rewrite, decompose, and decide whether to retrieve at all."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from tenantrag.text import normalize_query, tokenize

_SMALLTALK = re.compile(r"^(hi|hello|hey|thanks|thank you|namaste|good (morning|evening)|bye|ok|okay)\b[\s!.?]*$", re.I)
_MATH = re.compile(r"^[\d\s+\-*/().=?]+$")
_PRONOUN = re.compile(r"\b(it|that|this|they|them|its|their)\b", re.I)
_SPLIT = re.compile(r"\s+(?:and then|and also|as well as|; )\s+|\?\s+(?=\w)|\s+and\s+(?=(?:what|how|when|who|which)\b)", re.I)


@dataclass
class QueryPlan:
    original: str
    standalone: str
    route: str = "retrieve"  # retrieve | direct
    sub_queries: list[str] = field(default_factory=list)
    history_dependent: bool = False  # True -> bypass answer cache


def understand(query: str, history: list[str] | None = None) -> QueryPlan:
    q = normalize_query(query)
    if _SMALLTALK.match(q) or _MATH.match(q) or not tokenize(q):
        return QueryPlan(query, q, "direct")
    dep = False
    standalone = q
    if history and _PRONOUN.search(q) and len(tokenize(q)) <= 8:
        standalone = f"{normalize_query(history[-1])} {q}"  # carry the previous turn's subject forward
        dep = True
    parts = [p.strip(" ?.") for p in _SPLIT.split(standalone) if p and p.strip(" ?.")]
    subs = parts if len(parts) > 1 else []
    return QueryPlan(query, standalone, "retrieve", subs, dep)
