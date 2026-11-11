"""Per-tenant cost ledger. Token counts are approximated from words (x1.3) unless the LLM
returns usage; prices are PLACEHOLDERS, set them from your provider's price sheet."""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field

USD_PER_1K_PROMPT = 0.0003
USD_PER_1K_COMPLETION = 0.0012
USD_PER_EMBED_CALL = 0.00001


def approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3) + 1


@dataclass
class TenantUsage:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embed_calls: int = 0
    cache_hits: int = 0
    usd: float = 0.0
    saved_usd: float = 0.0


class CostLedger:
    def __init__(self):
        self._by: dict[str, TenantUsage] = defaultdict(TenantUsage)
        self._lock = threading.Lock()

    @staticmethod
    def price(prompt_tokens: int, completion_tokens: int, embed_calls: int) -> float:
        return (prompt_tokens / 1000 * USD_PER_1K_PROMPT + completion_tokens / 1000 * USD_PER_1K_COMPLETION
                + embed_calls * USD_PER_EMBED_CALL)

    def record(self, tenant_id: str, prompt_tokens: int, completion_tokens: int, embed_calls: int = 1,
               cache: str = "miss", saved_tokens: tuple[int, int] = (0, 0)) -> float:
        usd = self.price(prompt_tokens, completion_tokens, embed_calls)
        with self._lock:
            u = self._by[tenant_id]
            u.requests += 1
            u.prompt_tokens += prompt_tokens
            u.completion_tokens += completion_tokens
            u.embed_calls += embed_calls
            u.usd += usd
            if cache.startswith("hit"):
                u.cache_hits += 1
                u.saved_usd += self.price(*saved_tokens, 0)
        return usd

    def usage(self, tenant_id: str) -> TenantUsage:
        with self._lock:
            return self._by[tenant_id]

    def snapshot(self) -> dict[str, TenantUsage]:
        with self._lock:
            return dict(self._by)
