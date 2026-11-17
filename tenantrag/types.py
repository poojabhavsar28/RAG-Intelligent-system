"""Core value types shared by every TenantRAG module.

Principles (see docs/tenantrag/design.md):
  * Default deny: an ``Acl`` with no grants permits nobody.
  * Group and user ids are globally unique integers, so an ACL can never match
    across tenants even if a tenant filter is accidentally dropped.
  * ``Principal`` is built only from authenticated server-side identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Principal:
    tenant_id: str
    user_id: str | int
    group_ids: tuple[str | int, ...] = ()

    def fingerprint(self) -> tuple:
        return (self.tenant_id, tuple(sorted(self.group_ids)))


@dataclass(frozen=True, slots=True)
class Acl:
    is_public: bool = False
    allow_groups: frozenset[str | int] = frozenset()
    allow_users: frozenset[str | int] = frozenset()
    deny_groups: frozenset[str | int] = frozenset()
    version: int = 1

    def permits(self, p: Principal) -> bool:
        """Deny wins; then public; then group or user grants. Empty ACL = nobody."""
        if self.deny_groups.intersection(p.group_ids):
            return False
        if self.is_public:
            return True
        if self.allow_users and p.user_id in self.allow_users:
            return True
        return bool(self.allow_groups.intersection(p.group_ids))

    def with_version(self, version: int) -> "Acl":
        return Acl(self.is_public, self.allow_groups, self.allow_users, self.deny_groups, version)


@dataclass(slots=True)
class Chunk:
    tenant_id: str
    chunk_id: int
    doc_id: str
    version: int
    ordinal: int
    text: str
    lang: str = "en"
    acl: Acl = field(default_factory=Acl)  # denormalized copy; documents own the authority
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float
    source: str = "dense"  # dense | lexical | fused | rerank

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id
