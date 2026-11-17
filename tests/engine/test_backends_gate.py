import numpy as np
import pytest
from qdrant_client import QdrantClient

from tenantrag.corpus.simulator import build_corpus
from tenantrag.eval import gate
from tenantrag.tenancy.qdrant_backend import QdrantTenantStore
from tenantrag.text import HashingEmbedder


def test_qdrant_payload_multitenancy_never_leaks_and_matches_acl_semantics():
    corpus = build_corpus(6, max_docs=40, min_docs=10, queries_per_tenant=6, seed=17)
    emb = HashingEmbedder(64)
    store = QdrantTenantStore(QdrantClient(":memory:"), 64)
    store.ingest(corpus.chunks, emb.embed([c.text for c in corpus.chunks]))
    n = 0
    for g in corpus.golden:
        hits = store.dense(g.principal, emb.embed_query(g.query), 10)
        for h in hits:
            n += 1
            assert h.chunk.tenant_id == g.tenant_id
            assert corpus.docs[h.doc_id].acl.permits(g.principal)  # deny-wins / group / user / public all honoured
        if g.kind == "forbidden":
            assert g.forbidden_doc_id not in [h.doc_id for h in hits]
        else:
            assert g.relevant_doc_ids[0] in [h.doc_id for h in store.dense(g.principal, emb.embed_query(g.query), 40)]
    assert n > 0


def test_quality_gate_passes_against_committed_baseline_and_fails_on_regression(tmp_path, monkeypatch):
    assert gate.main([]) == 0
    bad = tmp_path / "b.json"
    bad.write_text('{"recall_at_1": 1.5, "recall_at_5": 1.5, "mrr": 1.5, "ndcg_at_10": 1.5}')
    monkeypatch.setattr(gate, "BASELINE", bad)
    assert gate.main([]) == 1  # a baseline the code can't meet => regression => CI fails
