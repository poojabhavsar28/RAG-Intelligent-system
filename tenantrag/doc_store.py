"""In-memory authoritative document store (the stand-in for the Postgres ``documents`` +
``document_acl`` tables). Documents own policy: ACL, lifecycle state, version, trust tier.
Chunks carry only denormalized copies, so changes here take effect on the next read."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from tenantrag.corpus.simulator import Corpus
from tenantrag.types import Acl


@dataclass
class DocStatus:
    doc_id: str
    state: str  # active | quarantined | deleted
    version_no: int
    acl: Acl
    trust_tier: int = 2
    injection_suspect: bool = False


class InMemoryDocStore:
    def __init__(self):
        self._docs: dict[tuple[str, str], DocStatus] = {}
        self._lock = threading.Lock()
        self.events: list[tuple[str, str, str]] = []  # (tenant, doc, kind)
        self.subscribers: list[Callable[[str, str, str], None]] = []
        self.deliver_events = True

    @classmethod
    def from_corpus(cls, corpus: Corpus) -> "InMemoryDocStore":
        s = cls()
        for d in corpus.docs.values():
            s._docs[(d.tenant_id, d.doc_id)] = DocStatus(d.doc_id, "active", 1, d.acl)
        return s

    def _emit(self, tenant: str, doc: str, kind: str) -> None:
        self.events.append((tenant, doc, kind))
        if self.deliver_events:
            for cb in self.subscribers:
                cb(tenant, doc, kind)

    # reads are tenant-keyed: a doc id from another tenant simply does not exist
    def status(self, tenant: str, doc_ids: list[str]) -> dict[str, DocStatus]:
        with self._lock:
            return {d: self._docs[(tenant, d)] for d in doc_ids if (tenant, d) in self._docs}

    def authority(self, tenant: str, doc_id: str) -> Acl | None:
        with self._lock:
            s = self._docs.get((tenant, doc_id))
            return s.acl if s and s.state == "active" else None

    def set_acl(self, tenant: str, doc_id: str, acl: Acl) -> int:
        with self._lock:
            s = self._docs[(tenant, doc_id)]
            s.acl = acl.with_version(s.acl.version + 1)
        self._emit(tenant, doc_id, "acl")
        return s.acl.version

    def soft_delete(self, tenant: str, doc_id: str) -> None:
        with self._lock:
            self._docs[(tenant, doc_id)].state = "deleted"
        self._emit(tenant, doc_id, "delete")

    def quarantine(self, tenant: str, doc_id: str) -> None:
        with self._lock:
            self._docs[(tenant, doc_id)].state = "quarantined"
        self._emit(tenant, doc_id, "quarantine")

    def bump_version(self, tenant: str, doc_id: str) -> int:
        with self._lock:
            s = self._docs[(tenant, doc_id)]
            s.version_no += 1
        self._emit(tenant, doc_id, "version")
        return s.version_no

    def mark_suspect(self, tenant: str, doc_id: str, suspect: bool = True) -> None:
        with self._lock:
            self._docs[(tenant, doc_id)].injection_suspect = suspect

    def direct_users(self, tenant: str) -> set[int]:
        """Users named in any document's allow_users: their cache scope must not be shared."""
        with self._lock:
            return {u for (t, _), s in self._docs.items() if t == tenant for u in s.acl.allow_users}
