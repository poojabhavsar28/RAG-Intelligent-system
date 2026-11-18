from __future__ import annotations

from typing import Sequence

from tenantrag.types import Chunk, Hit


def reciprocal_rank_fusion(rank_lists: Sequence[Sequence[Hit]], k: int = 60, top: int | None = None) -> list[Hit]:
    """RRF over any number of ranked lists, keyed by chunk id."""
    score: dict[int, float] = {}
    chunk: dict[int, Chunk] = {}
    for hits in rank_lists:
        for rank, h in enumerate(hits, 1):
            score[h.chunk.chunk_id] = score.get(h.chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunk[h.chunk.chunk_id] = h.chunk
    fused = [Hit(chunk[cid], s, "fused") for cid, s in sorted(score.items(), key=lambda kv: -kv[1])]
    return fused[:top] if top else fused
