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
from tenantrag.pipeline.llm import ExtractiveLLM, build_llm
from tenantrag.pipeline.orchestrator import RagPipeline
from tenantrag.retrieval.hybrid import HybridRetriever
from tenantrag.retrieval.rerank import OverlapReranker
from tenantrag.tenancy.router import TenantIndexRouter
from tenantrag.text import HashingEmbedder, build_embedder


def build():
    import os

    # Small deterministic demo corpus; use Ollama for both generation and embeddings by default.
    corpus = build_corpus(
        int(os.getenv("TENANTRAG_DEMO_TENANTS", "3")),
        max_docs=int(os.getenv("TENANTRAG_DEMO_MAX_DOCS", "12")),
        queries_per_tenant=2,
        seed=1,
    )
    # The demo remains deterministic/offline for tests. Production runs only
    # through tenantrag.api.production and always uses Ollama.
    use_ollama = os.getenv("TENANTRAG_DEMO_USE_OLLAMA", "false").lower() == "true"
    emb = build_embedder() if use_ollama else HashingEmbedder(128)
    llm = build_llm() if use_ollama else ExtractiveLLM()
    router = TenantIndexRouter("dedicated", emb)
    router.ingest(corpus.chunks)
    docs = InMemoryDocStore.from_corpus(corpus)
    cache = TenantScopedCache(MemoryBackend(), KeyService(), docs, emb, CacheConfig())
    docs.subscribers.append(cache.on_doc_event)
    retr = HybridRetriever(router, emb, reranker=OverlapReranker(), acl_authority=docs.authority)
    pipe = RagPipeline(retr, llm, cache=cache)
    tokens = {f"demo-{t}": corpus.users[t][0] for t in corpus.tenants}
    for tok, p in list(tokens.items())[:3]:
        print(f"demo token {tok} -> tenant {p.tenant_id}, user {p.user_id}")
    return create_app(pipe, Gateway(), static_token_resolver(tokens)), corpus


app, _corpus = build()
