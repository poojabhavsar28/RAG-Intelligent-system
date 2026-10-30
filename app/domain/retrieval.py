"""Tenant-aware retrieval contracts shared by vector-store backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class RetrievalContext:
    """Trusted identity and authorization context for one retrieval request."""

    tenant_id: str
    user_id: str | None = None
    group_ids: tuple[str, ...] = ()
    language: str = "en"
    require_document_acl: bool = True


@dataclass(frozen=True, slots=True)
class Hit:
    """A retrieved chunk with stable document and chunk references."""

    text: str
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    document_id: str | None = None
    chunk_id: str | None = None


class Retriever(Protocol):
    """Backend-independent interface for authorized retrieval."""

    def retrieve(
        self,
        context: RetrievalContext,
        query: str,
        k: int = 10,
    ) -> Sequence[Hit]:
        """Return at most k chunks authorized for the supplied context."""
        ...
