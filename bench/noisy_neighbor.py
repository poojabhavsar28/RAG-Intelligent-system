"""Noisy-neighbor experiment: tenant A floods the system while tenant B sends a steady trickle.

    python bench/noisy_neighbor.py

A fixed pool of workers (shared capacity) serves requests that take ~SERVICE_MS each. Compared:
  no_limits   everything is queued FIFO -> A's flood sits ahead of B (head-of-line blocking)
  gateway     per-tenant token bucket + in-flight cap reject A's excess with 429
Reports B's latency percentiles and B's error rate, and A's accepted/rejected counts.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tenantrag.eval.metrics import percentile
from tenantrag.gateway.limits import Gateway, TenantLimits

WORKERS, SERVICE_MS, DURATION_S = 4, 8.0, 4.0
A_RPS, B_RPS = 2000, 20


def run(use_gateway: bool) -> dict:
    gw = Gateway(TenantLimits(rate_per_s=60, burst=60, max_inflight=2)) if use_gateway else None
    pool = ThreadPoolExecutor(WORKERS)
    lat: dict[str, list[float]] = {"A": [], "B": []}
    rej = {"A": 0, "B": 0}
    ok = {"A": 0, "B": 0}
    lock = threading.Lock()

    def work(tenant: str, t_submit: float):
        time.sleep(SERVICE_MS / 1e3)
        with lock:
            lat[tenant].append((time.perf_counter() - t_submit) * 1e3)
            ok[tenant] += 1
        if gw:
            gw.release(tenant)

    def producer(tenant: str, rps: int):
        end = time.perf_counter() + DURATION_S
        gap = 1.0 / rps
        nxt = time.perf_counter()
        while time.perf_counter() < end:
            if gw:
                a = gw.admit(tenant)
                if not a.ok:
                    with lock:
                        rej[tenant] += 1
                    nxt += gap
                    time.sleep(max(0, nxt - time.perf_counter()))
                    continue
            pool.submit(work, tenant, time.perf_counter())
            nxt += gap
            time.sleep(max(0, nxt - time.perf_counter()))

    ts = [threading.Thread(target=producer, args=("A", A_RPS)), threading.Thread(target=producer, args=("B", B_RPS))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    pool.shutdown(wait=True)
    return {
        "config": "gateway" if use_gateway else "no_limits",
        "B_p50_ms": round(percentile(lat["B"], 50), 1), "B_p95_ms": round(percentile(lat["B"], 95), 1),
        "B_served": ok["B"], "B_rejected": rej["B"], "A_served": ok["A"], "A_rejected": rej["A"],
    }


if __name__ == "__main__":
    rows = [run(False), run(True)]
    Path("results").mkdir(exist_ok=True)
    Path("results/noisy_neighbor.json").write_text(json.dumps(rows, indent=2))
    cols = list(rows[0])
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    Path("results/noisy_neighbor.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
