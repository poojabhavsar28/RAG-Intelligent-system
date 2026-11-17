"""Small in-memory vector indexes used by the tenancy benchmark.

``IVFIndex`` is an inverted-file ANN index (k-means cells, ``nprobe`` cells
searched). It reproduces the *mechanism* by which filtered ANN loses recall:

  post-filter  fetch ``ef`` nearest candidates, THEN drop rows that fail the
               filter (how pgvector HNSW behaves without iterative scans).
               A selective filter leaves few or zero survivors.
  pre-filter   apply the filter while scanning the probed cells (filter-aware
               traversal, closer to Qdrant payload filtering). Recall now
               depends on whether the probed cells contain any allowed rows.

It is a model, not HNSW. ``bench/pg_filtered_recall.py`` measures the real
pgvector behaviour; the two are reported side by side.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

Mask = Callable[[np.ndarray], np.ndarray]


def _kmeans(x: np.ndarray, k: int, iters: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cent = x[rng.choice(len(x), size=k, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmax(x @ cent.T, axis=1)
        for c in range(k):
            m = assign == c
            if m.any():
                v = x[m].mean(axis=0)
                n = np.linalg.norm(v)
                cent[c] = v / n if n else v
    return cent


class FlatIndex:
    kind = "flat"

    def __init__(self, vecs: np.ndarray):
        self.vecs = np.ascontiguousarray(vecs, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.vecs)

    def search(self, q: np.ndarray, k: int, *, ef: int = 64, mask: Mask | None = None, mode: str = "post"):
        scores = self.vecs @ q
        if len(scores) == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        if mask is not None and mode == "pre":
            allidx = np.arange(len(scores))
            cand = allidx[mask(allidx)]
        else:
            n_c = min(max(ef, k), len(scores))
            cand = np.argpartition(-scores, n_c - 1)[:n_c] if len(scores) > n_c else np.arange(len(scores))
            if mask is not None and cand.size:
                cand = cand[mask(cand)]
        cand = cand[np.argsort(-scores[cand])][:k]
        return cand, scores[cand]

    @property
    def nbytes(self) -> int:
        return int(self.vecs.nbytes)


class IVFIndex:
    kind = "ivf"

    def __init__(self, vecs: np.ndarray, nlist: int | None = None, nprobe: int = 8, seed: int = 0):
        self.vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        n = len(self.vecs)
        self.nlist = nlist or max(2, int(np.sqrt(n)))
        self.nprobe = min(nprobe, self.nlist)
        self.centroids = _kmeans(self.vecs, self.nlist, seed=seed)
        assign = np.argmax(self.vecs @ self.centroids.T, axis=1)
        self.lists = [np.flatnonzero(assign == c) for c in range(self.nlist)]

    def __len__(self) -> int:
        return len(self.vecs)

    def search(self, q: np.ndarray, k: int, *, ef: int = 64, mask: Mask | None = None, mode: str = "post"):
        cells = np.argsort(-(self.centroids @ q))[: self.nprobe]
        members = np.concatenate([self.lists[c] for c in cells]) if len(cells) else np.array([], dtype=int)
        if members.size == 0:
            return members, np.array([], dtype=np.float32)
        if mask is not None and mode == "pre":
            members = members[mask(members)]
            if members.size == 0:
                return members, np.array([], dtype=np.float32)
            scores = self.vecs[members] @ q
        else:
            scores = self.vecs[members] @ q
            n_c = min(max(ef, k), len(scores))
            top = np.argpartition(-scores, n_c - 1)[:n_c] if len(scores) > n_c else np.arange(len(scores))
            members, scores = members[top], scores[top]
            if mask is not None and members.size:
                keep = mask(members)
                members, scores = members[keep], scores[keep]
        order = np.argsort(-scores)[:k]
        return members[order], scores[order]

    @property
    def nbytes(self) -> int:
        return int(self.vecs.nbytes + self.centroids.nbytes + sum(l.nbytes for l in self.lists))


def build_index(vecs: np.ndarray, *, flat_below: int = 512, nprobe: int = 8):
    return FlatIndex(vecs) if len(vecs) < flat_below else IVFIndex(vecs, nprobe=nprobe)
