"""Self-contained demo app: ``uvicorn tenantrag.api.demo:app``.
Builds a synthetic multi-tenant corpus, an in-memory pipeline with cache + gateway, and prints tokens."""
from __future__ import annotations

from tenantrag.cache.backend import MemoryBackend
from tenantrag.cache.keys import KeyService
from tenantrag.cache.scoped import CacheConfig, TenantScopedCache
from tenantrag.api.app import create_app, static_token_resolver
from tenantrag.corpus.simulator import build_corpus
from tenantrag.doc_store import InMemoryDocStore
from tenantrag.gateway.limits import Gateway
from tenantrag.pipeline.llm import ExtractiveLLM
from tenantrag.pipeline.orchestrator import RagPipeline
from tenantrag.retrieval.hybrid import HybridRetriever
from tenantrag.retrieval.rerank import OverlapReranker
from tenantrag.tenancy.router import TenantIndexRouter
from tenantrag.text import HashingEmbedder


def build():
    corpus = build_corpus(6, max_docs=40, queries_per_tenant=2, seed=1)
    emb = HashingEmbedder(128)
    router = TenantIndexRouter("dedicated", emb)
    router.ingest(corpus.chunks)
    docs = InMemoryDocStore.from_corpus(corpus)
    cache = TenantScopedCache(MemoryBackend(), KeyService(), docs, emb, CacheConfig())
    docs.subscribers.append(cache.on_doc_event)
    retr = HybridRetriever(router, emb, reranker=OverlapReranker(), acl_authority=docs.authority)
    pipe = RagPipeline(retr, ExtractiveLLM(), cache=cache)
    tokens = {f"demo-{t}": corpus.users[t][0] for t in corpus.tenants}
    for tok, p in list(tokens.items())[:3]:
        print(f"demo token {tok} -> tenant {p.tenant_id}, user {p.user_id}")
    return create_app(pipe, Gateway(), static_token_resolver(tokens)), corpus


app, _corpus = build()
