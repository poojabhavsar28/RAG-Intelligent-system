"""Integration tests for the PostgreSQL TenantRAG backend (needs POSTGRES_DATABASE_URL)."""
from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.infrastructure.postgres.tenant_db import create_postgres_engine, tenant_transaction
from tenantrag.corpus.simulator import build_corpus
from tenantrag.eval.harness import evaluate_retrieval
from tenantrag.tenancy.pg import PgHybridRetriever, PgTenantStore, doc_uuid
from tenantrag.text import HashingEmbedder
from tenantrag.types import Acl, Principal

pytestmark = pytest.mark.skipif(not os.getenv("POSTGRES_DATABASE_URL"), reason="POSTGRES_DATABASE_URL required")

N_TENANTS = 4


@pytest.fixture(scope="module")
def env():
    engine = create_postgres_engine(os.environ["POSTGRES_DATABASE_URL"])
    emb = HashingEmbedder(768)
    corpus = build_corpus(n_tenants=N_TENANTS, max_docs=40, min_docs=10, queries_per_tenant=6, seed=5)
    tenants = list(corpus.tenants)
    _cleanup(engine, tenants)
    store = PgTenantStore(engine, emb)
    store.ingest_corpus(corpus)
    yield engine, emb, corpus, store, PgHybridRetriever(engine, emb)
    _cleanup(engine, tenants)
    engine.dispose()


def _cleanup(engine, tenants):
    with engine.begin() as c:  # superuser/BYPASSRLS test connection
        c.execute(text("DELETE FROM tenants WHERE tenant_id = ANY(:t)"), {"t": tenants})


def test_hybrid_retrieval_quality_and_zero_leaks(env):
    engine, emb, corpus, store, retriever = env
    rep = evaluate_retrieval(corpus.golden, lambda g: retriever.retrieve(g.principal, g.query, 10))
    assert rep.leaks == 0
    assert rep.recall_at_5 >= 0.9, rep


def test_no_hit_ever_belongs_to_another_tenant(env):
    _, _, corpus, _, retriever = env
    for g in corpus.golden:
        for h in retriever.retrieve(g.principal, g.query, 20).hits:
            assert h.doc_id.startswith(g.tenant_id + "-doc-")


def test_dense_and_lexical_legs_both_work(env):
    _, _, corpus, _, retriever = env
    g = next(g for g in corpus.golden if g.kind == "answerable")
    dense = retriever.retrieve(g.principal, g.query, 5, lexical=False)
    lex = retriever.retrieve(g.principal, g.query, 5, dense=False)
    assert dense.hits and lex.hits
    assert g.relevant_doc_ids[0] in [h.doc_id for h in lex.hits]


def test_revocation_takes_effect_on_next_query_and_emits_event(env):
    _, _, corpus, store, retriever = env
    g = next(g for g in corpus.golden if g.kind == "answerable" and g.relevant_doc_ids)
    doc = g.relevant_doc_ids[0]
    assert doc in [h.doc_id for h in retriever.retrieve(g.principal, g.query, 10).hits]
    v = store.set_acl(g.tenant_id, doc, Acl())  # default deny: nobody
    assert v >= 2
    assert doc not in [h.doc_id for h in retriever.retrieve(g.principal, g.query, 10).hits]
    assert (str(doc_uuid(g.tenant_id, doc)), v) in store.pending_acl_events(g.tenant_id)
    store.set_acl(g.tenant_id, doc, Acl(is_public=True))  # restore for other tests
    assert doc in [h.doc_id for h in retriever.retrieve(g.principal, g.query, 10).hits]


def test_deny_overrides_public_and_group_grants(env):
    _, _, corpus, store, retriever = env
    g = next(g for g in corpus.golden if g.kind == "answerable")
    doc = g.relevant_doc_ids[0]
    group = g.principal.group_ids[0]
    store.set_acl(g.tenant_id, doc, Acl(is_public=True, allow_groups=frozenset({group}), deny_groups=frozenset({group})))
    assert doc not in [h.doc_id for h in retriever.retrieve(g.principal, g.query, 10).hits]
    store.set_acl(g.tenant_id, doc, Acl(is_public=True))


def test_soft_delete_hides_document_immediately(env):
    _, _, corpus, store, retriever = env
    g = [g for g in corpus.golden if g.kind == "answerable"][-1]
    doc = g.relevant_doc_ids[0]
    store.set_acl(g.tenant_id, doc, Acl(is_public=True))
    assert doc in [h.doc_id for h in retriever.retrieve(g.principal, g.query, 10).hits]
    store.soft_delete(g.tenant_id, doc)
    assert doc not in [h.doc_id for h in retriever.retrieve(g.principal, g.query, 10).hits]


def test_missing_acl_means_invisible(env):
    engine, emb, corpus, store, retriever = env
    g = next(g for g in corpus.golden if g.kind == "answerable" and g.relevant_doc_ids[0] in corpus.docs)
    doc = g.relevant_doc_ids[0]
    with tenant_transaction(engine, g.tenant_id) as conn:
        conn.execute(
            text("DELETE FROM document_acl WHERE tenant_id=:t AND document_id=:d"),
            {"t": g.tenant_id, "d": doc_uuid(g.tenant_id, doc)},
        )
    assert doc not in [h.doc_id for h in retriever.retrieve(g.principal, g.query, 20).hits]


def test_audit_log_records_hashes_not_query_text(env):
    engine, emb, corpus, _, _ = env
    g = next(g for g in corpus.golden if g.kind == "answerable")
    PgHybridRetriever(engine, emb, audit=True).retrieve(g.principal, g.query, 3)
    with tenant_transaction(engine, g.tenant_id) as conn:
        rows = conn.execute(text("SELECT query_hash, document_ids FROM audit_log")).all()
    assert rows and all(g.query not in str(r) for r in rows)


def test_new_tables_enforce_rls_for_non_superuser_role(env):
    engine, *_ = env
    admin_url = os.environ["POSTGRES_DATABASE_URL"]
    role, pw = "tenantrag_runtime_new", "ci-runtime-password"
    with engine.begin() as c:
        c.execute(text(
            f"DO $$ BEGIN CREATE ROLE {role} LOGIN PASSWORD '{pw}' NOSUPERUSER NOBYPASSRLS; "
            f"EXCEPTION WHEN duplicate_object THEN NULL; END $$"
        ))
        c.execute(text(f"GRANT USAGE ON SCHEMA public TO {role}"))
        c.execute(text(f"GRANT SELECT, INSERT, UPDATE, DELETE ON acl_events, audit_log, feedback, usage_ledger TO {role}"))
        c.execute(text(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {role}"))
        c.execute(text("INSERT INTO tenants (tenant_id, name) VALUES ('rls-new-a','a'),('rls-new-b','b') ON CONFLICT DO NOTHING"))
    url = make_url(admin_url).set(drivername="postgresql+psycopg", username=role, password=pw)
    rt = create_engine(url)
    try:
        with tenant_transaction(rt, "rls-new-a") as c:
            c.execute(text("INSERT INTO feedback (tenant_id, user_id, query, document_id, signal) VALUES ('rls-new-a','u','q','d',1)"))
        with tenant_transaction(rt, "rls-new-b") as c:
            assert c.execute(text("SELECT count(*) FROM feedback")).scalar_one() == 0
            with pytest.raises(Exception):  # WITH CHECK blocks writing another tenant's rows
                c.execute(text("INSERT INTO usage_ledger (tenant_id, request_id) VALUES ('rls-new-a','r')"))
        with rt.begin() as c:  # no tenant set: fails closed
            assert c.execute(text("SELECT count(*) FROM feedback")).scalar_one() == 0
    finally:
        rt.dispose()
        with engine.begin() as c:
            c.execute(text("DELETE FROM tenants WHERE tenant_id IN ('rls-new-a','rls-new-b')"))


# Unique indexes that intentionally do NOT lead with tenant_id (hashed credentials looked up
# before the tenant is known, plus surrogate keys). Everything else must lead with tenant_id,
# otherwise a unique-violation error could reveal that another tenant holds a value.
ALLOWED_GLOBAL_UNIQUE = {
    "tenants_pkey", "app_users_pkey", "sessions_pkey", "sessions_token_hash_key", "customer_tokens_pkey",
    "chat_history_pkey", "qa_templates_pkey", "acl_events_pkey", "audit_log_pkey", "feedback_pkey",
    "usage_ledger_pkey", "alembic_version_pkc",
}


def test_unique_indexes_lead_with_tenant_id(env):
    engine, *_ = env
    with engine.connect() as c:
        rows = c.execute(text(
            "SELECT i.indexname, i.indexdef FROM pg_indexes i WHERE i.schemaname='public' AND i.indexdef ILIKE '%UNIQUE%'"
        )).all()
    offenders = []
    for name, ddl in rows:
        cols = ddl.split("(", 1)[1].rsplit(")", 1)[0].split(",")[0].strip()
        if cols != "tenant_id" and name not in ALLOWED_GLOBAL_UNIQUE and not name.startswith("auth_token"):
            offenders.append((name, ddl))
    assert not offenders, offenders
