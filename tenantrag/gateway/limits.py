"""Gateway admission control: per-tenant token bucket (rate), daily quota, and an in-flight
cap so one tenant cannot occupy all workers. State is keyed by tenant id only."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class TenantLimits:
    rate_per_s: float = 20.0
    burst: int = 40
    max_inflight: int = 8
    daily_request_quota: int = 100_000
    daily_usd_quota: float = 50.0


@dataclass
class Admission:
    ok: bool
    reason: str = ""
    retry_after_s: float = 0.0
    status: int = 200


class _Bucket:
    def __init__(self, rate: float, burst: int):
        self.rate, self.burst = rate, burst
        self.tokens, self.t = float(burst), time.monotonic()

    def take(self) -> float:
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
        self.t = now
        if self.tokens >= 1:
            self.tokens -= 1
            return 0.0
        return (1 - self.tokens) / self.rate


class Gateway:
    def __init__(self, default: TenantLimits | None = None, overrides: dict[str, TenantLimits] | None = None):
        self.default, self.overrides = default or TenantLimits(), overrides or {}
        self._b: dict[str, _Bucket] = {}
        self._inflight: dict[str, int] = {}
        self._reqs: dict[str, int] = {}
        self._usd: dict[str, float] = {}
        self._lock = threading.Lock()

    def limits(self, tenant_id: str) -> TenantLimits:
        return self.overrides.get(tenant_id, self.default)

    def admit(self, tenant_id: str) -> Admission:
        lim = self.limits(tenant_id)
        with self._lock:
            if self._reqs.get(tenant_id, 0) >= lim.daily_request_quota or self._usd.get(tenant_id, 0.0) >= lim.daily_usd_quota:
                return Admission(False, "quota_exceeded", 3600.0, 429)
            if self._inflight.get(tenant_id, 0) >= lim.max_inflight:
                return Admission(False, "too_many_in_flight", 0.05, 429)
            b = self._b.setdefault(tenant_id, _Bucket(lim.rate_per_s, lim.burst))
            wait = b.take()
            if wait > 0:
                return Admission(False, "rate_limited", wait, 429)
            self._inflight[tenant_id] = self._inflight.get(tenant_id, 0) + 1
            self._reqs[tenant_id] = self._reqs.get(tenant_id, 0) + 1
        return Admission(True)

    def release(self, tenant_id: str, usd: float = 0.0) -> None:
        with self._lock:
            self._inflight[tenant_id] = max(0, self._inflight.get(tenant_id, 1) - 1)
            self._usd[tenant_id] = self._usd.get(tenant_id, 0.0) + usd
