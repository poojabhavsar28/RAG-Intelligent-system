import pytest

from tenantrag.corpus.simulator import build_corpus
from tenantrag.doc_store import InMemoryDocStore
from tenantrag.eval.harness import evaluate_retrieval
from tenantrag.retrieval.hybrid import HybridRetriever
from tenantrag.retrieval.rerank import OverlapReranker
from tenantrag.tenancy.router import TenantIndexRouter
from tenantrag.text import HashingEmbedder
from tenantrag.types import Acl, Principal


@pytest.fixture(scope="module")
def world():
    corpus = build_corpus(12, max_docs=60, min_docs=8, queries_per_tenant=6, seed=9)
    return corpus, HashingEmbedder(128)


@pytest.mark.parametrize("mode", ["dedicated", "shared", "hybrid"])
@pytest.mark.parametrize("filter_mode", ["post", "pre"])
def test_zero_cross_tenant_and_cross_user_leaks_in_every_tenancy_mode(world, mode, filter_mode):
    corpus, emb = world
    router = TenantIndexRouter(mode, emb, filter_mode=filter_mode, big_tenant_chunks=60, ef=16)  # tiny ef = worst case
    router.ingest(corpus.chunks)
    ds = InMemoryDocStore.from_corpus(corpus)
    r = HybridRetriever(router, emb, reranker=OverlapReranker(), acl_authority=ds.authority)
    rep = evaluate_retrieval(corpus.golden, lambda g: r.retrieve(g.principal, g.query, 10))
    assert rep.leaks == 0 and rep.tenant_violations == 0
    for g in corpus.golden:  # every returned doc is in the caller's tenant AND permitted
        for h in r.retrieve(g.principal, g.query, 10).hits:
            assert h.chunk.tenant_id == g.tenant_id
            assert corpus.docs[h.doc_id].acl.permits(g.principal)


def test_attacker_principal_with_foreign_tenant_gets_nothing_from_others(world):
    corpus, emb = world
    router = TenantIndexRouter("shared", emb)
    router.ingest(corpus.chunks)
    victim = corpus.golden[0]
    ghost = Principal("tenant-does-not-exist", 1, ())
    r = HybridRetriever(router, emb)
    assert r.retrieve(ghost, victim.query, 10).hits == []


def test_revocation_enforced_immediately_but_grant_waits_for_propagation(world):
    corpus, emb = world
    router = TenantIndexRouter("dedicated", emb)
    router.ingest(corpus.chunks)
    ds = InMemoryDocStore.from_corpus(corpus)
    r = HybridRetriever(router, emb, acl_authority=ds.authority)
    g = next(g for g in corpus.golden if g.kind == "answerable")
    doc = g.relevant_doc_ids[0]
    assert doc in [h.doc_id for h in r.retrieve(g.principal, g.query, 10).hits]
    ds.set_acl(g.tenant_id, doc, Acl())  # revoke; chunk copies are STALE (still allow)
    res = r.retrieve(g.principal, g.query, 10)
    assert doc not in [h.doc_id for h in res.hits]
    assert res.stale_acl_hits >= 1  # revocation caught by the authoritative check, and counted
    # grant to a user whose chunk copy still says "denied": fail-safe, not visible yet
    denied = next(x for x in corpus.golden if x.kind == "forbidden" and x.tenant_id == g.tenant_id)
    d2 = denied.forbidden_doc_id
    ds.set_acl(g.tenant_id, d2, Acl(is_public=True))
    assert d2 not in [h.doc_id for h in r.retrieve(denied.principal, denied.query, 10).hits]
    for c in corpus.chunks:  # propagation worker updates the denormalized copies
        if c.doc_id == d2:
            c.acl = Acl(is_public=True)
    assert d2 in [h.doc_id for h in r.retrieve(denied.principal, denied.query, 10).hits]


def test_deleted_and_quarantined_docs_vanish_on_next_request(world):
    corpus, emb = world
    router = TenantIndexRouter("dedicated", emb)
    router.ingest(corpus.chunks)
    ds = InMemoryDocStore.from_corpus(corpus)
    r = HybridRetriever(router, emb, acl_authority=ds.authority)
    gs = [g for g in corpus.golden if g.kind == "answerable"][:2]
    ds.soft_delete(gs[0].tenant_id, gs[0].relevant_doc_ids[0])
    ds.quarantine(gs[1].tenant_id, gs[1].relevant_doc_ids[0])
    for g in gs:
        assert g.relevant_doc_ids[0] not in [h.doc_id for h in r.retrieve(g.principal, g.query, 10).hits]


def test_missing_authority_record_means_invisible(world):
    corpus, emb = world
    router = TenantIndexRouter("dedicated", emb)
    router.ingest(corpus.chunks)
    r = HybridRetriever(router, emb, acl_authority=lambda t, d: None)
    g = corpus.golden[0]
    assert r.retrieve(g.principal, g.query, 10).hits == []


def test_audit_log_records_hashes_only(world):
    from tenantrag.security.audit import AuditLog

    corpus, emb = world
    router = TenantIndexRouter("dedicated", emb)
    router.ingest(corpus.chunks)
    log = AuditLog()
    r = HybridRetriever(router, emb, audit=log)
    g = corpus.golden[0]
    r.retrieve(g.principal, g.query, 5)
    es = log.entries_for(g.tenant_id)
    assert es and all(g.query not in repr(e) for e in es)
    assert log.entries_for("other-tenant") == []
