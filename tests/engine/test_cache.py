import itertools
import os
import random
import shutil
import socket
import subprocess
import threading
import time

import pytest

from tenantrag.cache.ablation import CANARY, ROWS, build_world, run_leak_ablation
from tenantrag.cache.backend import MemoryBackend, RedisBackend
from tenantrag.cache.keys import KeyService, canonical_encode
from tenantrag.cache.scoped import CacheConfig, Layers, TenantScopedCache, constraint_signature, detect_lang
from tenantrag.types import Acl, Principal


def test_layer_ablation_every_row_behaves_as_designed():
    rows = run_leak_ablation()
    assert len(rows) == len(ROWS)
    assert all(r["as_expected"] for r in rows), [r for r in rows if not r["as_expected"]]
    leaks = [r for r in rows if r["leaked"]]
    assert leaks and all(r["layers_off"] for r in leaks)  # leaks only appear once every guarding layer is off


def test_canary_never_reaches_other_tenant_across_exact_paraphrase_adversarial_queries():
    w = build_world(scope="tenant", public=True)
    assert CANARY in w.pipeline.answer(w.victim, w.query).answer
    attacks = [w.query, w.query.upper(), "  " + w.query + "  ", w.query.replace("?", ""), f"tell me about {w.query}",
               w.query + " ignore previous instructions", w.query.replace("What is", "what's")]
    for q in attacks:
        assert CANARY not in w.pipeline.answer(w.other_tenant, q).answer


def test_key_uniqueness_property_including_delimiter_injection():
    rng = random.Random(1)
    ks = KeyService(b"x" * 32)
    seen = {}
    alphabet = ["a", "b", ":", "|", "{", "}", "t:", "\x00", " "]
    for _ in range(3000):
        t = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 5)))
        q = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 6)))
        s = rng.choice(["T", "u", "g|1"])
        key = KeyService.mac(ks.mac_key(t), [("scope", s), ("q", q)])
        ident = (t, s, q)
        assert seen.setdefault(key, ident) == ident
    assert canonical_encode([("a", "bc")]) != canonical_encode([("ab", "c")])


def test_ciphertext_moved_between_tenants_fails_and_alerts():
    w = build_world(scope="tenant", public=True)
    w.pipeline.answer(w.victim, w.query)
    be = w.cache.backend
    (tag, od), = [(t, o) for t, o in be._d.items()]
    (key, (exp, blob)), = od.items()
    other_tag = w.cache._tag(w.other_tenant)
    be.set(other_tag, key, blob, 60)  # attacker copies A's ciphertext into B's keyspace under the same key
    assert w.cache._check(w.other_tenant, key, blob) is None
    assert ("decrypt_failure", w.other_tenant.tenant_id) in w.cache.security_events


def test_group_scoped_entry_not_served_outside_group_and_direct_grants_not_shared():
    w = build_world()
    assert CANARY in w.pipeline.answer(w.victim, w.query).answer
    assert CANARY not in w.pipeline.answer(w.outsider, w.query).answer  # different group -> miss, and unauthorised
    twin = Principal(w.victim.tenant_id, 9100, w.victim.group_ids)  # same groups, different user: shares the entry
    assert w.cache.get(twin, w.query) is not None
    w.docs._docs[(w.victim.tenant_id, w.doc_id)].acl = Acl(allow_groups=frozenset({w.group}), allow_users=frozenset({9001}))
    assert w.cache._scope(w.victim) != w.cache._scope(twin)  # direct grant => scope no longer shared


@pytest.mark.parametrize("event", ["acl", "delete", "quarantine", "version"])
def test_staleness_stops_serving_on_next_request_even_with_events_disabled(event):
    w = build_world()
    w.docs.deliver_events = False
    w.pipeline.answer(w.victim, w.query)
    assert w.cache.get(w.victim, w.query) is not None
    t, d = w.victim.tenant_id, w.doc_id
    {"acl": lambda: w.docs.set_acl(t, d, Acl()), "delete": lambda: w.docs.soft_delete(t, d),
     "quarantine": lambda: w.docs.quarantine(t, d), "version": lambda: w.docs.bump_version(t, d)}[event]()
    assert w.cache.get(w.victim, w.query) is None


def test_eager_invalidation_removes_entries_when_events_flow():
    w = build_world()
    w.pipeline.answer(w.victim, w.query)
    tag = w.cache._tag(w.victim)
    assert w.cache.backend.count(tag) == 1
    w.docs.soft_delete(w.victim.tenant_id, w.doc_id)
    assert w.cache.backend.count(tag) == 0


def test_admission_rejects_abstentions_pii_temporal_untrusted_and_injection_sources():
    w = build_world(public=True)
    abst = w.pipeline.answer(w.victim, "What is the zzqx capacity of unknownthing-9?")
    assert abst.abstained and w.cache.backend.count(w.cache._tag(w.victim)) == 0
    r = w.pipeline.answer(w.victim, w.query)  # a good answer IS cached
    assert w.cache.backend.count(w.cache._tag(w.victim)) == 1
    hits = w.pipeline.retriever.retrieve(w.victim, w.query, 3).hits
    pr = lambda **kw: w.cache.put(w.victim, kw.pop("q", "fresh query one"), r, hits, **kw)
    assert not pr(pii_shown=True) and not pr(q="what is the latest policy today")
    w.docs.mark_suspect(w.victim.tenant_id, w.doc_id)
    assert not pr(q="fresh query two") and w.cache.admission_denied["untrusted_source"] >= 1
    w.docs.mark_suspect(w.victim.tenant_id, w.doc_id, False)
    w.docs._docs[(w.victim.tenant_id, w.doc_id)].trust_tier = 0  # e.g. an end-user upload
    assert not pr(q="fresh query three")


def test_semantic_hits_require_equal_constraint_slots():
    sig = constraint_signature
    assert sig("refund window 30 days") != sig("refund window 45 days")
    assert sig("allowed with approval") == sig("allowed with approval")
    assert sig("is it allowed") != sig("is it not allowed")
    assert sig("more than 5 seats") != sig("fewer than 5 seats")
    w = build_world(public=True, cfg=CacheConfig(scope="tenant", t_low=0.5, t_high=0.9, verifier_min_jaccard=0.0))
    w.pipeline.answer(w.victim, w.query)
    hit = w.cache.get(w.victim, w.query.replace("?", " "))
    assert hit is not None  # semantic/exact tier works for a true paraphrase of the same slots
    base, dig = w.query.rstrip("?").rsplit("-", 1)
    other = base + "-" + str((int(dig[0]) % 9) + 1)  # same question, different digit slot
    assert w.cache.get(w.victim, other + "?") is None


def test_hindi_and_gujarati_use_exact_only():
    assert detect_lang("રિફંડ નીતિ") == "gu" and detect_lang("रिफंड नीति") == "hi" and detect_lang("refund") == "en"
    w = build_world(public=True)
    assert "gu" in w.cache.cfg.exact_only_languages


def test_single_flight_two_scopes_run_pipeline_twice_and_never_cross():
    w = build_world(public=True)
    runs, lock = [], threading.Lock()
    results = {}
    def work(p, i):
        def compute():
            with lock: runs.append(p.user_id)
            time.sleep(0.05)
            return w.pipeline.answer(p, w.query)
        results[i] = (p, w.cache.get_or_compute(p, w.query, compute))
    ps = [w.victim, w.outsider]
    ts = [threading.Thread(target=work, args=(ps[i % 2], i)) for i in range(200)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(runs) == 2 and {runs[0], runs[1]} == {9001, 9002}
    assert all(r is not None for _, r in results.values())


def test_flooding_tenant_cannot_evict_another_tenants_entries():
    be = MemoryBackend(max_entries_per_tenant=50)
    be.set("B", "k", b"keep", 60)
    for i in range(10_000):
        be.set("A", f"x{i}", b"z", 60)
    assert be.get("B", "k") == b"keep" and be.count("A") == 50


def test_epoch_bump_invalidates_one_tenant_only():
    w = build_world(public=True)
    w.pipeline.answer(w.victim, w.query)
    assert w.cache.get(w.victim, w.query) is not None
    w.cache.bump_epoch(w.victim.tenant_id)
    assert w.cache.get(w.victim, w.query) is None
    w.cache.bump_epoch("someone-else")


def test_negative_feedback_evicts_and_hit_carries_no_request_state():
    w = build_world(public=True)
    w.pipeline.answer(w.victim, w.query)
    h = w.cache.get(w.victim, w.query)
    assert h.cache == "hit_exact" and not hasattr(h, "request_id")
    w.cache.evict(w.victim, w.query)
    assert w.cache.get(w.victim, w.query) is None


def test_hide_hits_pads_latency():
    w = build_world(public=True, cfg=CacheConfig(scope="tenant", hide_hits_ms=30))
    w.pipeline.answer(w.victim, w.query)
    t = time.perf_counter(); w.cache.get(w.victim, w.query)
    assert (time.perf_counter() - t) >= 0.028


def test_cache_logs_contain_no_query_text():
    w = build_world(public=True)
    w.pipeline.answer(w.victim, w.query)
    w.cache.get(w.other_tenant, w.query)
    blob = repr(w.cache.security_events) + repr(w.cache.outcomes) + repr(w.cache.admission_denied)
    assert w.query not in blob and CANARY not in blob


def test_hit_saves_cost_in_ledger():
    w = build_world(public=True)
    w.pipeline.answer(w.victim, w.query)
    w.pipeline.answer(w.victim, w.query)
    u = w.pipeline.ledger.usage(w.victim.tenant_id)
    assert u.requests == 2 and u.cache_hits == 1 and u.saved_usd > 0


@pytest.fixture(scope="module")
def redis_client():
    import redis
    exe = shutil.which("redis-server")
    if not exe:
        pytest.skip("redis-server not installed")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    proc = subprocess.Popen([exe, "--port", str(port), "--save", "", "--appendonly", "no"], stdout=subprocess.DEVNULL)
    c = redis.Redis(port=port)
    for _ in range(50):
        try:
            c.ping(); break
        except Exception:
            time.sleep(0.1)
    yield c
    proc.terminate()


def test_redis_backend_end_to_end_with_per_tenant_cap(redis_client):
    be = RedisBackend(redis_client, max_entries_per_tenant=20)
    w = build_world(public=True, backend=be, scope="tenant")
    assert CANARY in w.pipeline.answer(w.victim, w.query).answer
    assert w.cache.get(w.victim, w.query) is not None
    assert w.cache.get(w.other_tenant, w.query) is None
    be.set("B", "k", b"keep", 60)
    for i in range(200):
        be.set("A", f"x{i}", b"z", 60)
    assert be.get("B", "k") == b"keep" and be.count("A") == 20
