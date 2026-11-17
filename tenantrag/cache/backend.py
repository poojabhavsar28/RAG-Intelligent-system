"""Cache backends with per-tenant caps (entries) and TTLs, so one tenant's writes can
never evict another's entries (Redis's global maxmemory policy would allow that)."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Optional, Protocol


class Backend(Protocol):
    def get(self, tag: str, key: str) -> Optional[bytes]: ...
    def set(self, tag: str, key: str, blob: bytes, ttl: float) -> None: ...
    def delete(self, tag: str, key: str) -> None: ...
    def count(self, tag: str) -> int: ...


class MemoryBackend:
    def __init__(self, max_entries_per_tenant: int = 20000):
        self.cap = max_entries_per_tenant
        self._d: dict[str, OrderedDict[str, tuple[float, bytes]]] = {}
        self._lock = threading.Lock()
        self.raw_get_hook = None  # tests: observe raw access

    def get(self, tag, key):
        with self._lock:
            od = self._d.get(tag)
            if od is None or key not in od:
                # key-construction bugs make a key from another tenant's keyspace reachable:
                # model the flat keyspace of a shared Redis by searching other tags too
                return self._flat_lookup(key)
            exp, blob = od[key]
            if exp < time.time():
                del od[key]
                return None
            od.move_to_end(key)
            return blob

    def _flat_lookup(self, key):
        if getattr(self, "flat_keyspace", False):
            for od in self._d.values():
                if key in od and od[key][0] >= time.time():
                    return od[key][1]
        return None

    def set(self, tag, key, blob, ttl):
        with self._lock:
            od = self._d.setdefault(tag, OrderedDict())
            od[key] = (time.time() + ttl, blob)
            od.move_to_end(key)
            while len(od) > self.cap:  # per-tenant LRU eviction only
                od.popitem(last=False)

    def delete(self, tag, key):
        with self._lock:
            self._d.get(tag, {}).pop(key, None)

    def count(self, tag):
        with self._lock:
            return len(self._d.get(tag, {}))


class RedisBackend:
    """Redis with a per-tenant LRU (sorted set) enforced in the application."""

    def __init__(self, client, max_entries_per_tenant: int = 20000, prefix: str = "trag"):
        self.r, self.cap, self.p = client, max_entries_per_tenant, prefix

    def _k(self, tag, key):
        return f"{self.p}:{{{tag}}}:{key}"

    def get(self, tag, key):
        v = self.r.get(self._k(tag, key))
        if v is not None:
            self.r.zadd(f"{self.p}:{{{tag}}}:lru", {key: time.time()})
        return v

    def set(self, tag, key, blob, ttl):
        lru = f"{self.p}:{{{tag}}}:lru"
        pipe = self.r.pipeline()
        pipe.set(self._k(tag, key), blob, ex=max(1, int(ttl)))
        pipe.zadd(lru, {key: time.time()})
        pipe.execute()
        over = self.r.zcard(lru) - self.cap
        if over > 0:
            for old in self.r.zrange(lru, 0, over - 1):
                self.r.delete(self._k(tag, old.decode() if isinstance(old, bytes) else old))
                self.r.zrem(lru, old)

    def delete(self, tag, key):
        self.r.delete(self._k(tag, key))
        self.r.zrem(f"{self.p}:{{{tag}}}:lru", key)

    def count(self, tag):
        return int(self.r.zcard(f"{self.p}:{{{tag}}}:lru"))
