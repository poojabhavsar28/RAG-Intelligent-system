"""Tenant-scoped answer cache (exact + semantic) that is structurally hard to leak.

Independent layers; each would stop a leak even if the others failed:
  1 typed access        callers hold a TenantScopedCache; no method accepts a raw key
  2 key derivation      HMAC with a per-tenant secret, length-prefixed fields, tenant tag
  3 keyspace/index      per-tenant keyspace + per-tenant semantic index (flat exact scan, scope mask)
  4 AEAD                AES-GCM, per-tenant data key, AAD = tenant|key|schema (``Layers.aead``)
  5 tenant tag          tenant id inside the sealed envelope, checked after decrypt (``Layers.tag_check``)
  6 validate-on-read    doc state/version/trust/ACL re-checked for EVERY context doc (``Layers.validate``)
  7 re-materialization  cited docs re-checked against the live ACL at emission (``Layers.rematerialize``)

``bugs`` injects deliberate faults so tests can show which layer catches each one.
A semantic cache can also return a *wrong but non-leaking* answer; that is handled by the
constraint signature, two thresholds and a paraphrase check (see ``_semantic_lookup``).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from tenantrag.cache.backend import Backend
from tenantrag.cache.keys import KeyService
from tenantrag.pipeline.orchestrator import PipelineResponse
from tenantrag.security.guards import check_output, is_suspect
from tenantrag.text import Embedder, normalize_query, tokenize
from tenantrag.types import Hit, Principal

SCHEMA = "sc/3"
_TEMPORAL = {"today", "latest", "current", "currently", "now", "tonight", "yesterday", "tomorrow", "recent"}
_SIG_WORDS = {"not", "no", "without", "never", "more", "less", "fewer", "than", "least", "most", "before",
              "after", "all", "any", "only", "except", "नहीं", "बिना", "नथી", "વગર"}


@dataclass
class CacheConfig:
    mode: str = "exact+semantic"  # off | exact | exact+semantic
    scope: str = "acl_group"  # user | acl_group | tenant
    t_high: float = 0.97  # placeholders: calibrate per embedding model and tenant
    t_low: float = 0.90
    verifier_min_jaccard: float = 0.75
    exact_only_languages: tuple[str, ...] = ("gu",)
    min_support_rate: float = 0.95
    min_source_trust: int = 1
    skip_temporal: bool = True
    popularity_min: int = 1
    ttl: float = 6 * 3600
    hide_hits_ms: float = 0.0  # pad hit latency toward the miss path (timing side channel)
    semantic_cap: int = 20000


@dataclass
class Layers:
    aead: bool = True
    tag_check: bool = True
    validate: bool = True
    rematerialize: bool = True


def detect_lang(q: str) -> str:
    for ch in q:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:
            return "hi"
        if 0x0A80 <= o <= 0x0AFF:
            return "gu"
    return "en"


def constraint_signature(q: str) -> str:
    toks = tokenize(q)
    slots = sorted({t for t in toks if any(c.isdigit() for c in t) or t in _SIG_WORDS})
    return hashlib.sha256("|".join(slots).encode()).hexdigest()[:16]


def _jaccard(a: str, b: str) -> float:
    x, y = set(tokenize(a)), set(tokenize(b))
    return len(x & y) / len(x | y) if x | y else 0.0


class _Sem:
    """Per-tenant flat semantic index. Scope is applied as a mask BEFORE scoring."""

    def __init__(self):
        self.vecs: list[np.ndarray] = []
        self.meta: list[dict] = []

    def add(self, vec, meta):
        self.vecs.append(vec)
        self.meta.append(meta)

    def drop_key(self, key):
        keep = [i for i, m in enumerate(self.meta) if m["key"] != key]
        self.vecs = [self.vecs[i] for i in keep]
        self.meta = [self.meta[i] for i in keep]

    def topk(self, q, k, scope, tenant_tag=None):
        idx = [i for i, m in enumerate(self.meta) if m["scope"] == scope and (tenant_tag is None or m["tag"] == tenant_tag)]
        if not idx:
            return []
        sims = np.stack([self.vecs[i] for i in idx]) @ q
        order = np.argsort(-sims)[:k]
        return [(float(sims[o]), self.meta[idx[o]]) for o in order]


class TenantScopedCache:
    def __init__(self, backend: Backend, keys: KeyService, docs, embedder: Embedder | None = None,
                 cfg: CacheConfig | None = None, layers: Layers | None = None,
                 bugs: frozenset[str] = frozenset(), canaries: tuple[str, ...] = ()):
        self.backend, self.keys, self.docs, self.embedder = backend, keys, docs, embedder
        self.cfg, self.layers, self.bugs, self.canaries = cfg or CacheConfig(), layers or Layers(), bugs, canaries
        self._epoch: dict[str, int] = {}
        self._sem: dict[str, _Sem] = {}
        self._rev: dict[tuple[str, str], set[str]] = {}
        self._seen: dict[tuple[str, str], int] = {}
        self._flights: dict[str, threading.Lock] = {}
        self._fl_lock = threading.Lock()
        self.security_events: list[tuple[str, str]] = []
        self.outcomes: dict[str, int] = {}
        self.admission_denied: dict[str, int] = {}

    # ---------------- keys / scope ----------------
    def _scope(self, p: Principal) -> str:
        if "omit_scope" in self.bugs:
            return "T"
        mk = self.keys.mac_key(p.tenant_id)
        mode = self.cfg.scope
        if mode == "tenant":
            return "T"
        if mode == "user":
            return KeyService.mac(mk, [("u", p.user_id)])
        direct = p.user_id if p.user_id in self.docs.direct_users(p.tenant_id) else None
        return KeyService.mac(mk, [("g", sorted(p.group_ids)), ("direct", direct)])

    def _tag(self, p: Principal) -> str:
        return "shared" if "omit_tenant_key" in self.bugs else self.keys.tenant_tag(p.tenant_id)

    def _key(self, p: Principal, qnorm: str) -> str:
        fields = [("schema", SCHEMA), ("scope", self._scope(p)), ("q", qnorm),
                  ("epoch", self._epoch.get(p.tenant_id, 0))]
        if "omit_tenant_key" in self.bugs:
            return "ac:" + KeyService.mac(self.keys.global_mac_key(), fields)
        return "ac:" + KeyService.mac(self.keys.mac_key(p.tenant_id), fields)

    def bump_epoch(self, tenant_id: str) -> None:
        """Kill switch: invalidates every entry for a tenant immediately."""
        self._epoch[tenant_id] = self._epoch.get(tenant_id, 0) + 1
        self._sem.pop(tenant_id, None)

    # ---------------- seal / open ----------------
    def _aad(self, tenant_id: str, key: str) -> bytes:
        return f"{tenant_id}|{hashlib.sha256(key.encode()).hexdigest()[:16]}|{SCHEMA}".encode()

    def _seal(self, p: Principal, key: str, entry: dict) -> bytes:
        data = json.dumps(entry).encode()
        if not self.layers.aead:
            return b"P" + data
        nonce = os.urandom(12)
        return b"E" + nonce + AESGCM(self.keys.data_key(p.tenant_id)).encrypt(nonce, data, self._aad(p.tenant_id, key))

    def _open(self, p: Principal, key: str, blob: bytes) -> dict | None:
        try:
            if blob[:1] == b"P":
                return json.loads(blob[1:])
            nonce, ct = blob[1:13], blob[13:]
            data = AESGCM(self.keys.data_key(p.tenant_id)).decrypt(nonce, ct, self._aad(p.tenant_id, key))
            return json.loads(data)
        except (InvalidTag, ValueError, KeyError):
            self.security_events.append(("decrypt_failure", p.tenant_id))
            return None

    # ---------------- lookup ----------------
    def get(self, p: Principal, query: str) -> PipelineResponse | None:
        if self.cfg.mode == "off":
            return None
        t0 = time.perf_counter()
        qn = normalize_query(query)
        res = self._exact(p, qn)
        label = "hit_exact"
        if res is None and self.cfg.mode == "exact+semantic" and self.embedder is not None:
            res, label = self._semantic_lookup(p, qn), "hit_semantic"
        if res is None:
            self.outcomes["miss"] = self.outcomes.get("miss", 0) + 1
            return None
        self.outcomes[label] = self.outcomes.get(label, 0) + 1
        out = self._materialize(p, res, label)
        if out is None:
            return None
        if self.cfg.hide_hits_ms:
            left = self.cfg.hide_hits_ms / 1e3 - (time.perf_counter() - t0)
            if left > 0:
                time.sleep(left)
        return out

    def _exact(self, p, qn) -> dict | None:
        key = self._key(p, qn)
        blob = self.backend.get(self._tag(p), key)
        return self._check(p, key, blob) if blob else None

    def _check(self, p, key, blob) -> dict | None:
        entry = self._open(p, key, blob)
        if entry is None:
            return None
        if self.layers.tag_check and entry.get("tenant_id") != p.tenant_id:
            self.security_events.append(("tenant_tag_mismatch", p.tenant_id))  # two layers failed: page someone
            return None
        if self.layers.validate and not self._validate(p, entry):
            return None
        if time.time() > entry["expires"]:
            return None
        return entry

    def _validate(self, p: Principal, entry: dict) -> bool:
        ids = [d["doc_id"] for d in entry["context_docs"]]
        rows = self.docs.status(p.tenant_id, ids)
        for d in entry["context_docs"]:
            r = rows.get(d["doc_id"])
            if r is None or r.state != "active" or r.version_no != d["version"]:
                return False
            if r.trust_tier < self.cfg.min_source_trust or r.injection_suspect:
                return False
            if not r.acl.permits(p):
                return False
        return True

    def _materialize(self, p: Principal, entry: dict, label: str) -> PipelineResponse | None:
        if self.layers.rematerialize:
            for doc in entry["citations"]:  # ACL check at emission against the live authority
                acl = self.docs.authority(p.tenant_id, doc)
                if acl is None or not acl.permits(p):
                    return None
            chk = check_output(entry["answer"], {f"d{i}" for i in range(1, entry["n_labels"] + 1)},
                               canaries=self.canaries, require_citation=False)
            if not chk.ok:
                return None
        r = PipelineResponse(entry["answer"], list(entry["citations"]), False, None, "retrieve", 0,
                             [d["doc_id"] for d in entry["context_docs"]], label)
        r.prompt_tokens, r.completion_tokens = entry["prompt_tokens"], entry["completion_tokens"]  # SAVED tokens
        r.support_rate = entry["support_rate"]
        return r  # fresh per-request fields are built by the caller; nothing request-specific is cached

    # ---------------- semantic tier ----------------
    def _index(self, tenant_id: str) -> _Sem:
        k = "*" if "shared_semantic" in self.bugs else tenant_id
        return self._sem.setdefault(k, _Sem())

    def _semantic_lookup(self, p: Principal, qn: str) -> dict | None:
        lang = detect_lang(qn)
        toks = set(tokenize(qn))
        if lang in self.cfg.exact_only_languages or (self.cfg.skip_temporal and toks & _TEMPORAL):
            return None
        qv = self.embedder.embed_query(qn)
        sig = constraint_signature(qn)
        shared = "shared_semantic" in self.bugs
        for sim, m in self._index(p.tenant_id).topk(qv, 3, self._scope(p), None if shared else self._tag(p)):
            if sim < self.cfg.t_low:
                break
            if m["sig"] != sig:  # slots (numbers, negations, comparators) must be equal
                continue
            if sim < self.cfg.t_high and _jaccard(qn, m["qnorm"]) < self.cfg.verifier_min_jaccard:
                continue
            blob = self.backend.get(m["tag"] if shared else self._tag(p), m["key"])
            if not blob:
                continue
            entry = self._check(p, m["key"], blob)
            if entry:
                return entry
        return None

    # ---------------- admission ----------------
    def _deny(self, why: str) -> bool:
        self.admission_denied[why] = self.admission_denied.get(why, 0) + 1
        return False

    def put(self, p: Principal, query: str, resp: PipelineResponse, ctx_hits: list[Hit], pii_shown: bool = False) -> bool:
        if self.cfg.mode == "off":
            return False
        qn = normalize_query(query)
        if resp.abstained or resp.blocked or resp.route != "retrieve":
            return self._deny("not_answered")
        if resp.support_rate < self.cfg.min_support_rate:
            return self._deny("low_support")
        if pii_shown:
            return self._deny("pii")
        if self.cfg.skip_temporal and set(tokenize(qn)) & _TEMPORAL:
            return self._deny("temporal")
        if any(is_suspect(h.chunk.text) for h in ctx_hits):
            return self._deny("injection_suspect")
        doc_ids = list(dict.fromkeys(h.doc_id for h in ctx_hits))
        rows = self.docs.status(p.tenant_id, doc_ids)
        if len(rows) != len(doc_ids) or any(r.trust_tier < self.cfg.min_source_trust or r.injection_suspect for r in rows.values()):
            return self._deny("untrusted_source")
        key = self._key(p, qn)
        seen = self._seen[(p.tenant_id, key)] = self._seen.get((p.tenant_id, key), 0) + 1
        if seen < self.cfg.popularity_min:
            return self._deny("popularity")
        entry = {
            "tenant_id": p.tenant_id, "query_norm": qn, "sig": constraint_signature(qn),
            "answer": resp.answer, "citations": resp.citations, "n_labels": len(ctx_hits),
            "context_docs": [{"doc_id": d, "version": rows[d].version_no} for d in doc_ids],
            "support_rate": resp.support_rate, "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens, "expires": time.time() + self.cfg.ttl,
            "origin": hashlib.sha256(f"{p.user_id}|{time.time_ns()}".encode()).hexdigest()[:16],
        }
        tag = self._tag(p)
        self.backend.set(tag, key, self._seal(p, key, entry), self.cfg.ttl)
        for d in doc_ids:
            self._rev.setdefault((p.tenant_id, d), set()).add(key)
        if self.cfg.mode == "exact+semantic" and self.embedder is not None and detect_lang(qn) not in self.cfg.exact_only_languages:
            idx = self._index(p.tenant_id)
            if len(idx.meta) < self.cfg.semantic_cap:
                idx.add(self.embedder.embed_query(qn), {"key": key, "scope": self._scope(p), "sig": entry["sig"],
                                                         "qnorm": qn, "tag": tag})
        return True

    # ---------------- invalidation ----------------
    def on_doc_event(self, tenant_id: str, doc_id: str, kind: str) -> None:
        """Eager invalidation (memory hygiene). Correctness never depends on it: validate-on-read does."""
        for key in self._rev.pop((tenant_id, doc_id), set()):
            self.backend.delete(self._tag(Principal(tenant_id, 0)), key)
            self._index(tenant_id).drop_key(key)

    def evict(self, p: Principal, query: str) -> None:
        key = self._key(p, normalize_query(query))
        self.backend.delete(self._tag(p), key)
        self._index(p.tenant_id).drop_key(key)

    # ---------------- single flight ----------------
    def get_or_compute(self, p: Principal, query: str, compute):
        """Coalesce concurrent identical requests. The lock key includes the scope, so a
        waiter can never receive a result generated for a different scope."""
        hit = self.get(p, query)
        if hit is not None:
            return hit
        fk = self._tag(p) + self._key(p, normalize_query(query))
        with self._fl_lock:
            lock = self._flights.setdefault(fk, threading.Lock())
        with lock:
            hit = self.get(p, query)  # waiters re-check; they still pass validate + materialize
            return hit if hit is not None else compute()
