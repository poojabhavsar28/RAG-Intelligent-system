import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tenantrag.adaptation.tenant_reranker import AdaptiveReranker, Feedback, FeedbackStore
from tenantrag.api.app import create_app, static_token_resolver
from tenantrag.api.demo import build
from tenantrag.corpus.simulator import build_corpus
from tenantrag.gateway.limits import Gateway, TenantLimits
from tenantrag.retrieval.hybrid import HybridRetriever
from tenantrag.tenancy.router import TenantIndexRouter
from tenantrag.text import HashingEmbedder


def test_rate_limit_inflight_quota_and_tenant_independence():
    gw = Gateway(TenantLimits(rate_per_s=1000, burst=3, max_inflight=2, daily_request_quota=5))
    assert gw.admit("a").ok and gw.admit("a").ok
    r = gw.admit("a")
    assert not r.ok and r.reason == "too_many_in_flight" and r.status == 429
    assert gw.admit("b").ok  # b unaffected by a's saturation
    gw.release("a"); gw.release("a")
    gw2 = Gateway(TenantLimits(rate_per_s=0.5, burst=2, max_inflight=99))
    assert gw2.admit("x").ok and gw2.admit("x").ok
    r = gw2.admit("x")
    assert not r.ok and r.reason == "rate_limited" and r.retry_after_s > 0
    gw3 = Gateway(TenantLimits(daily_request_quota=1, max_inflight=9))
    assert gw3.admit("q").ok and gw3.admit("q").reason == "quota_exceeded"
    gw4 = Gateway(TenantLimits(daily_usd_quota=0.01, max_inflight=9))
    gw4.admit("z"); gw4.release("z", usd=0.02)
    assert gw4.admit("z").reason == "quota_exceeded"


def test_per_tenant_overrides():
    gw = Gateway(TenantLimits(max_inflight=1), {"big": TenantLimits(max_inflight=5, burst=100, rate_per_s=100)})
    assert gw.admit("small").ok and not gw.admit("small").ok
    assert all(gw.admit("big").ok for _ in range(5))


def test_adaptation_cold_start_keeps_shared_model_and_feedback_is_tenant_scoped():
    c = build_corpus(2, max_docs=40, min_docs=20, queries_per_tenant=30, seed=3)
    emb = HashingEmbedder(128)
    router = TenantIndexRouter("dedicated", emb); router.ingest(c.chunks)
    ad = AdaptiveReranker(min_examples=20)
    retr = HybridRetriever(router, emb, reranker=ad)
    store = FeedbackStore()
    g = [x for x in c.golden if x.kind == "answerable" and x.tenant_id == "tenant-0000"]
    for x in g[:3]:
        store.add("tenant-0000", Feedback(x.query, x.relevant_doc_ids[0], 1, True))
    cand = lambda q: retr.retrieve(g[0].principal, q, 10, rerank=False).hits
    assert ad.train("tenant-0000", store, cand) is False and not ad.is_adapted("tenant-0000")  # cold start
    for x in g:
        store.add("tenant-0000", Feedback(x.query, x.relevant_doc_ids[0], 1, True))
    assert ad.train("tenant-0000", store, cand) is True and ad.is_adapted("tenant-0000")
    assert not ad.is_adapted("tenant-0001") and store.for_tenant("tenant-0001") == []
    h = retr.retrieve(g[0].principal, g[0].query, 5).hits  # adapted model never reintroduces unauthorised hits
    assert all(x.chunk.tenant_id == "tenant-0000" for x in h)


@pytest.fixture(scope="module")
def client():
    app, corpus = build()
    return TestClient(app), corpus


def test_api_requires_auth(client):
    c, _ = client
    assert c.post("/v2/ask", json={"query": "x"}).status_code == 401
    assert c.post("/v2/ask", json={"query": "x"}, headers={"Authorization": "Bearer nope"}).status_code == 401


def test_api_answer_has_no_internal_fields_and_cache_is_invisible(client):
    c, corpus = client
    g = next(g for g in corpus.golden if g.kind == "answerable")
    tok = {"Authorization": f"Bearer demo-{g.tenant_id}"}
    # the demo token maps to the tenant's first user; make sure the doc is visible to that user
    u = corpus.users[g.tenant_id][0]
    d = next(d for d in corpus.docs.values() if d.tenant_id == g.tenant_id and d.acl.permits(u))
    q = f"What is the {d.attr} for {d.subject}?"
    r1 = c.post("/v2/ask", json={"query": q}, headers=tok)
    r2 = c.post("/v2/ask", json={"query": q}, headers=tok)  # served from cache
    assert r1.status_code == r2.status_code == 200
    assert set(r1.json()) == {"request_id", "answer", "citations", "abstained"} and set(r2.json()) == set(r1.json())
    assert r1.json()["answer"] == r2.json()["answer"] and r1.json()["request_id"] != r2.json()["request_id"]
    assert d.value in r1.json()["answer"]


def test_api_tenant_cannot_see_other_tenants_content_or_usage(client):
    c, corpus = client
    t0, t1 = list(corpus.tenants)[:2]
    d = next(d for d in corpus.docs.values() if d.tenant_id == t0 and d.acl.is_public)
    r = c.post("/v2/ask", json={"query": f"What is the {d.attr} for {d.subject}?"},
               headers={"Authorization": f"Bearer demo-{t1}"})
    assert d.doc_id not in r.json()["citations"]
    u = c.get("/v2/usage", headers={"Authorization": f"Bearer demo-{t1}"}).json()
    assert u["tenant_id"] == t1


def test_api_rate_limit_returns_429_with_retry_after():
    corpus = build_corpus(2, max_docs=10, min_docs=6, queries_per_tenant=1, seed=2)
    t = list(corpus.tenants)[0]
    gw = Gateway(TenantLimits(rate_per_s=0.2, burst=1, max_inflight=9))
    tc = TestClient(create_app(_pipeline(), gw, static_token_resolver({"k": corpus.users[t][0]})))
    h = {"Authorization": "Bearer k"}
    assert tc.post("/v2/ask", json={"query": "hello"}, headers=h).status_code == 200
    r = tc.post("/v2/ask", json={"query": "hello"}, headers=h)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def _pipeline():
    from tenantrag.cache.backend import MemoryBackend
    from tenantrag.cache.keys import KeyService
    from tenantrag.cache.scoped import CacheConfig, TenantScopedCache
    from tenantrag.doc_store import InMemoryDocStore
    from tenantrag.pipeline.llm import ExtractiveLLM
    from tenantrag.pipeline.orchestrator import RagPipeline
    c = build_corpus(2, max_docs=10, min_docs=6, queries_per_tenant=1, seed=2)
    emb = HashingEmbedder(128)
    r = TenantIndexRouter("dedicated", emb); r.ingest(c.chunks)
    ds = InMemoryDocStore.from_corpus(c)
    return RagPipeline(HybridRetriever(r, emb, acl_authority=ds.authority), ExtractiveLLM())


def test_feedback_endpoint_stores_per_tenant_and_evicts_bad_cached_answer(client):
    c, corpus = client
    t = list(corpus.tenants)[2]
    u = corpus.users[t][0]
    d = next(d for d in corpus.docs.values() if d.tenant_id == t and d.acl.permits(u))
    q = f"What is the {d.attr} for {d.subject}?"
    h = {"Authorization": f"Bearer demo-{t}"}
    c.post("/v2/ask", json={"query": q}, headers=h)
    assert c.post("/v2/feedback", json={"query": q, "document_id": d.doc_id, "signal": -1}, headers=h).status_code == 204
    assert c.get("/v2/usage", headers=h).json()["cache_hits"] == 0
    c.post("/v2/ask", json={"query": q}, headers=h)
    assert c.get("/v2/usage", headers=h).json()["cache_hits"] == 0  # evicted, so the next ask was a miss
