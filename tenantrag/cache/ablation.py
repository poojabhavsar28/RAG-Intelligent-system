"""Layer-ablation leak experiment: inject one bug at a time into a test build, plant a canary in
the victim's cached answer, attack as another tenant / unauthorised user, and record whether the
canary leaks. Disabling layers one by one shows each layer is independently load-bearing."""
from __future__ import annotations

from dataclasses import dataclass

from tenantrag.cache.backend import MemoryBackend
from tenantrag.cache.keys import KeyService
from tenantrag.cache.scoped import CacheConfig, Layers, TenantScopedCache
from tenantrag.corpus.simulator import build_corpus
from tenantrag.doc_store import InMemoryDocStore
from tenantrag.pipeline.llm import ExtractiveLLM
from tenantrag.pipeline.orchestrator import RagPipeline
from tenantrag.retrieval.hybrid import HybridRetriever
from tenantrag.retrieval.rerank import OverlapReranker
from tenantrag.tenancy.router import TenantIndexRouter
from tenantrag.text import HashingEmbedder
from tenantrag.types import Acl, Principal

CANARY = "CANARYZX9QK"


@dataclass
class World:
    cache: TenantScopedCache
    pipeline: RagPipeline
    docs: InMemoryDocStore
    corpus: object
    victim: Principal
    outsider: Principal  # same tenant, no access to the document
    other_tenant: Principal
    query: str
    doc_id: str
    group: int


def build_world(*, layers: Layers | None = None, bugs: frozenset = frozenset(), scope: str = "acl_group",
                public: bool = False, cfg: CacheConfig | None = None, backend=None, seed: int = 21) -> World:
    corpus = build_corpus(2, max_docs=20, min_docs=10, queries_per_tenant=1, seed=seed)
    ta, tb = list(corpus.tenants)
    d = next(x for x in corpus.docs.values() if x.tenant_id == ta)
    g1, g2 = corpus.tenants[ta]["groups"][:2]
    acl = Acl(is_public=True) if public else Acl(allow_groups=frozenset({g1}))
    d.acl = acl
    for c in corpus.chunks:
        if c.doc_id == d.doc_id:
            c.acl = acl
            c.text = c.text.replace(d.value, CANARY)
    emb = HashingEmbedder(128)
    router = TenantIndexRouter("dedicated", emb)
    router.ingest(corpus.chunks)
    ds = InMemoryDocStore.from_corpus(corpus)
    cache = TenantScopedCache(backend or MemoryBackend(), KeyService(b"k" * 32), ds, emb,
                              cfg or CacheConfig(scope=scope), layers, bugs)
    ds.subscribers.append(cache.on_doc_event)
    retr = HybridRetriever(router, emb, reranker=OverlapReranker(), acl_authority=ds.authority)
    pipe = RagPipeline(retr, ExtractiveLLM(), cache=cache)
    return World(cache, pipe, ds, corpus, Principal(ta, 9001, (g1,)), Principal(ta, 9002, (g2,)),
                 Principal(tb, 9003, ()), f"What is the {d.attr} for {d.subject}?", d.doc_id, g1)


def _leaked(resp) -> bool:
    return CANARY in resp.answer


def _scenario(kind: str, layers: Layers, bugs: frozenset) -> tuple[bool, str]:
    if kind == "cross_tenant":
        w = build_world(layers=layers, bugs=bugs, scope="tenant", public=True)
        assert CANARY in w.pipeline.answer(w.victim, w.query).answer  # victim populates the cache
        r = w.pipeline.answer(w.other_tenant, w.query)
    elif kind == "cross_user":
        w = build_world(layers=layers, bugs=bugs)
        assert CANARY in w.pipeline.answer(w.victim, w.query).answer
        r = w.pipeline.answer(w.outsider, w.query)
    elif kind == "stale_acl":
        w = build_world(layers=layers, bugs=bugs)
        w.docs.deliver_events = False  # invalidation events lost: correctness must not depend on them
        assert CANARY in w.pipeline.answer(w.victim, w.query).answer
        w.docs.set_acl(w.victim.tenant_id, w.doc_id, Acl())  # revoke everyone
        r = w.pipeline.answer(w.victim, w.query)
    else:
        raise ValueError(kind)
    ev = ",".join(sorted({e for e, _ in w.cache.security_events})) or "-"
    return _leaked(r), ev


ROWS = [
    # name, scenario, bug, layers disabled, expected leak, layer expected to stop it
    ("no bug (control)", "cross_tenant", set(), set(), False, "1-3 key/keyspace isolation"),
    ("key omits tenant", "cross_tenant", {"omit_tenant_key"}, set(), False, "4 AEAD"),
    ("key omits tenant, AEAD off", "cross_tenant", {"omit_tenant_key"}, {"aead"}, False, "5 tenant tag"),
    ("key omits tenant, AEAD+tag off", "cross_tenant", {"omit_tenant_key"}, {"aead", "tag_check"}, False, "6 validate-on-read"),
    ("key omits tenant, AEAD+tag+validate off", "cross_tenant", {"omit_tenant_key"}, {"aead", "tag_check", "validate"}, False, "7 re-materialization"),
    ("key omits tenant, layers 4-7 off", "cross_tenant", {"omit_tenant_key"}, {"aead", "tag_check", "validate", "rematerialize"}, True, "none (expected leak)"),
    ("scope omitted from key", "cross_user", {"omit_scope"}, set(), False, "6 validate-on-read"),
    ("scope omitted, validate off", "cross_user", {"omit_scope"}, {"validate"}, False, "7 re-materialization"),
    ("scope omitted, validate+remat off", "cross_user", {"omit_scope"}, {"validate", "rematerialize"}, True, "none (expected leak)"),
    ("shared semantic index, filter dropped", "cross_tenant", {"shared_semantic"}, set(), False, "2/4-7"),
    ("ACL revoked, events lost", "stale_acl", set(), set(), False, "6 validate-on-read"),
    ("ACL revoked, events lost, validate off", "stale_acl", set(), {"validate"}, False, "7 re-materialization"),
    ("ACL revoked, events lost, validate+remat off", "stale_acl", set(), {"validate", "rematerialize"}, True, "none (expected leak)"),
]


def run_leak_ablation() -> list[dict]:
    out = []
    for name, kind, bugs, off, expected, layer in ROWS:
        layers = Layers(**{k: False for k in off})
        leaked, ev = _scenario(kind, layers, frozenset(bugs))
        out.append({"injected_bug": name, "attack": kind, "layers_off": ",".join(sorted(off)) or "-",
                    "leaked": leaked, "expected_leak": expected, "stopped_by": layer, "security_events": ev,
                    "as_expected": leaked == expected})
    return out


def to_markdown(rows: list[dict]) -> str:
    cols = ["injected_bug", "attack", "layers_off", "leaked", "stopped_by", "security_events", "as_expected"]
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(md)
