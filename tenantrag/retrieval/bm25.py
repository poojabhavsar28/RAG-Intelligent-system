"""Per-tenant BM25. Statistics (IDF, average length) are computed per tenant, so a
tenant's ranking never depends on, or leaks, another tenant's term statistics."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Sequence

from tenantrag.text import tokenize
from tenantrag.types import Chunk


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.4, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lens: list[int] = []
        for i, c in enumerate(self.chunks):
            toks = tokenize(c.text)
            self.lens.append(len(toks))
            tf: dict[str, int] = defaultdict(int)
            for t in toks:
                tf[t] += 1
            for t, n in tf.items():
                self.post[t].append((i, n))
        self.n = len(self.chunks)
        self.avg = (sum(self.lens) / self.n) if self.n else 0.0

    def search(self, query: str, k: int, allow: Callable[[Chunk], bool] | None = None):
        scores: dict[int, float] = defaultdict(float)
        for t in set(tokenize(query)):
            plist = self.post.get(t)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, tf in plist:
                if allow is not None and not allow(self.chunks[i]):  # ACL pre-filter: exact
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg or 1))
                scores[i] += idf * tf * (self.k1 + 1) / denom
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.chunks[i], s) for i, s in top]
