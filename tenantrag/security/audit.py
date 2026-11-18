"""Append-only audit log of who retrieved which chunks.

Stores only hashes of the query (never query text) plus chunk/doc ids, so the log
itself cannot become a leak of tenant content. ``entries_for`` is tenant-scoped.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from tenantrag.types import Principal


@dataclass(frozen=True)
class AuditEntry:
    ts: float
    tenant_id: str
    user_id: int
    query_hash: str
    chunk_ids: tuple[int, ...]
    doc_ids: tuple[str, ...]


class AuditLog:
    def __init__(self, path: str | Path | None = None):
        self._entries: list[AuditEntry] = []
        self._lock = threading.Lock()
        self._path = Path(path) if path else None

    def record(self, p: Principal, query_hash: str, chunk_ids: Sequence[int], doc_ids: Sequence[str]) -> None:
        e = AuditEntry(time.time(), p.tenant_id, p.user_id, query_hash, tuple(chunk_ids), tuple(doc_ids))
        with self._lock:
            self._entries.append(e)
            if self._path:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(e)) + "\n")

    def entries_for(self, tenant_id: str) -> list[AuditEntry]:
        with self._lock:
            return [e for e in self._entries if e.tenant_id == tenant_id]

    def __len__(self) -> int:
        return len(self._entries)
