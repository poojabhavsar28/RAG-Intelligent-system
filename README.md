# TenantRAG

**How do you serve hundreds of tenants with isolated, permission-aware, continuously evaluated retrieval, and what does each design choice cost in recall, latency and money?**

TenantRAG is a multi-tenant RAG engine built to answer that question with measurements instead of opinions. Every claim below is backed by a script in `bench/` or a test in `tests/engine/`, and the numbers in this README are copied from `results/`.

> The original single-tenant chatbot (`app/`, `ai_assistant.py`, `main.py`) is untouched except for one security fix (see *Changes to legacy code*). The new engine lives in `tenantrag/`.

## Answer up front

| Question | Finding (measured here) |
|---|---|
| Which tenancy model? | **Per-tenant indexes** keep recall exact (1.0) but cost ~**260x** more per tenant in the cost model at 1,000 tenants. **A shared HNSW index with a tenant filter loses most of its recall** on real pgvector (0.27 → 0.11 recall@10 at `ef_search=40` from 100 to 1,000 tenants). **Hybrid (big tenants dedicated)** does not rescue the long tail. |
| Can BM25 hide the problem? | Yes. With the lexical leg on, every mode reaches 0.99–1.0 recall@5 on this corpus. Dense-only numbers expose the filter effect; fused numbers can mask it. Report both. |
| Does the production query use HNSW at all? | **No, not at this size.** EXPLAIN shows the planner drives from `documents` (tenant + ACL) and exact-sorts the tenant's chunks, so recall is 1.0 by construction. The filtered-ANN loss appears once a tenant's partition is large enough for the planner to pick HNSW. |
| Can a cache leak across tenants? | Each injected bug is stopped by the expected layer. A leak appears **only when every guarding layer is switched off** (13-row ablation below). Leak rate with all layers on: **0**. |
| Does self-correction reduce hallucination? | On unanswerable questions, hallucination drops **99% → 1%**, answer accuracy rises **92% → 100%** (typo repair), and p95 latency rises ~**1.5 ms → 3.0 ms**. (Deterministic extractive LLM; see caveats.) |
| Does per-tenant adaptation help? | **Not on this corpus.** Three attempts, none positive (see *What didn't work*). |
| Noisy neighbor? | Without limits tenant B's p50 is **6,180 ms** behind a flooder; with per-tenant limits it is **8.2 ms**. |

Zero cross-tenant and cross-user leaks were observed in every configuration run (all tenancy modes × pre/post filtering, all ablations).

## Architecture

```
Client → Gateway (auth → Principal, token bucket, in-flight cap, quota)
       → Cache lookup (exact, then semantic; validate-on-read)           tenantrag/cache
       → Query understanding (route: retrieve or not, rewrite, decompose) tenantrag/pipeline/understanding.py
       → Retrieval: dense + BM25 → RRF → authoritative ACL check → rerank tenantrag/retrieval, tenancy
       → Self-check loop (judge → repair query → retry → abstain)         tenantrag/pipeline/selfcheck.py
       → Generation with fenced context + citations                       tenantrag/pipeline/llm.py
       → Guardrails (citation check, canary, PII)                         tenantrag/security/guards.py
       → Response (no internals) + cost ledger + audit log
Ingestion: parse → chunk → embed → index per tenant → ACLs   (tenantrag/tenancy/pg.py, router.py)
Eval: golden sets, metrics, CI gate                           (tenantrag/eval, eval/baseline.json)
```

Design notes are in [`docs/tenantrag/design.md`](docs/tenantrag/design.md); the Postgres foundation (RLS, roles, partitions) is in [`docs/tenantrag/postgres-foundation.md`](docs/tenantrag/postgres-foundation.md).

## Results

All data is **synthetic** (client data cannot be published): skewed tenant sizes, realistic ACL shapes, and subject names deliberately reused across tenants so a shared index holds many near-duplicates. The embedder is a hashing n-gram model, so results measure *isolation and index behaviour*, not embedding quality.

### 1. Tenancy models on real pgvector (HNSW m=16, ef_construction=64, pgvector 0.6.0)
`bench/pg_filtered_recall.py` — recall@10 versus exact tenant-scoped top-10.

| tenants | chunks | mode | ef_search | recall@10 | empty results | p95 ms | index MB | relations |
|---|---|---|---|---|---|---|---|---|
| 100 | 5,724 | dedicated | 40 | **1.00** | 0 | 1.55 | 53.3 | 100 |
| 100 | 5,724 | hybrid | 40 | 0.317 | 1 | 1.47 | 47.9 | 4 |
| 100 | 5,724 | shared | 40 | 0.272 | 2 | 1.78 | 47.7 | 1 |
| 100 | 5,724 | shared | 100 | 0.372 | 0 | 2.01 | 47.7 | 1 |
| 100 | 5,724 | shared | 400 | 0.488 | 0 | 4.81 | 47.7 | 1 |
| 100 | 5,724 | shared | 1000 | 0.657 | 0 | 7.78 | 47.7 | 1 |
| 1000 | 21,924 | dedicated | 40 | **1.00** | 0 | 1.44 | 237.7 | 1000 |
| 1000 | 21,924 | hybrid | 40 | 0.112 | 20 | 2.21 | 182.6 | 4 |
| 1000 | 21,924 | shared | 40 | 0.106 | 19 | 3.10 | 182.4 | 1 |
| 1000 | 21,924 | shared | 100 | 0.145 | 3 | 3.27 | 182.4 | 1 |
| 1000 | 21,924 | shared | 400 | 0.223 | 0 | 7.87 | 182.4 | 1 |
| 1000 | 21,924 | shared | 1000 | 0.270 | 0 | 18.77 | 182.4 | 1 |

pgvector applies the tenant predicate *after* the HNSW scan returns `ef_search` candidates, so selective filters return few or zero rows (≈9% of 1,000-tenant probes returned nothing at the default `ef_search`). Raising `ef_search` recovers recall slowly while latency grows ~6x. Iterative scans (pgvector ≥ 0.8) are the intended mitigation and are **not** covered here; the sandbox had 0.6.0. `enable_seqscan=off` was used to force the planner onto HNSW for this experiment.

### 2. Tenancy models in the engine's own indexes (in-memory IVF model, ACLs, fused retrieval)
`bench/tenancy_scaling.py` — an IVF model reproduces the post-filter vs pre-filter mechanism; cost uses editable assumptions ($6/GB-month RAM, 4 MiB fixed overhead per index).

| tenants | mode | filter | search effort | dense recall@5 | hybrid recall@5 | dense p95 ms | indexes | modeled $/tenant/month |
|---|---|---|---|---|---|---|---|---|
| 10 | dedicated | n/a | ef64/np8 | 0.929 | 1.0 | 0.24 | 10 | 0.0243 |
| 10 | shared | post | ef64/np8 | 0.750 | 0.893 | 0.22 | 1 | 0.0032 |
| 10 | hybrid | post | ef64/np8 | 0.821 | 0.982 | 0.24 | 4 | 0.0102 |
| 100 | dedicated | n/a | ef64/np8 | 0.990 | 0.995 | 0.26 | 100 | 0.0236 |
| 100 | shared | post | ef64/np8 | 0.852 | 0.990 | 0.22 | 1 | 0.0004 |
| 1000 | dedicated | n/a | ef64/np8 | 1.000 | 1.0 | 0.21 | 1000 | 0.0235 |
| 1000 | shared | post | ef64/np8 | 0.781 | 1.0 | 0.37 | 1 | 0.00009 |
| 1000 | shared | pre | ef64/np8 | 0.886 | 1.0 | 0.53 | 1 | 0.00009 |
| 1000 | shared | post | ef256/np16 | 0.900 | 1.0 | 0.51 | 1 | 0.00009 |
| 1000 | shared | post | ef1024/np32 | 0.952 | 1.0 | 0.92 | 1 | 0.00009 |
| 1000 | shared | post | exact scan | 1.000 | 1.0 | 5.94 | 1 | 0.00009 |

Full table (incl. hybrid at each scale and pre-filter rows): `results/tenancy_scaling.md`. Takeaway: shared indexes are ~260x cheaper per tenant, dedicated indexes buy exactness, and the gap can be closed on the shared index by paying search effort (latency) or by using filter-aware traversal.

### 3. Retrieval ablation (`bench/ablations.py`)
| config | recall@1 | recall@5 | MRR | nDCG@10 | p95 ms | leaks |
|---|---|---|---|---|---|---|
| dense only | 0.735 | 0.974 | 0.827 | 0.865 | 0.21 | 0 |
| BM25 only | 0.901 | 1.000 | 0.945 | 0.959 | 0.29 | 0 |
| hybrid (RRF) | 0.861 | 0.980 | 0.916 | 0.934 | 0.77 | 0 |
| hybrid + rerank | **0.921** | 0.993 | **0.955** | **0.966** | 2.09 | 0 |

Honest note: plain RRF hybrid is *worse* than BM25 alone here, because the hashing embedder adds noise to a corpus where exact-token matches dominate. The reranker recovers it. With a real semantic embedder expect different ordering.

### 4. Self-correcting loop (`bench/ablations.py`, 151 answerable / 99 unanswerable / 80 forbidden)
| config | answer accuracy | wrongly abstained | hallucination on unanswerable | forbidden doc leaked | p95 ms |
|---|---|---|---|---|---|
| no self-check | 0.921 | 0.000 | **0.990** | 0 | 1.51 |
| judge + abstain (1 attempt) | 0.669 | 0.331 | 0.010 | 0 | 1.68 |
| judge + repair + retry (2 attempts) | **1.000** | 0.000 | 0.010 | 0 | 3.01 |

Abstaining alone kills hallucination but refuses a third of answerable questions (typos). Repairing the query against the retrieved vocabulary and retrying fixes that.

### 5. Cache isolation: layer-ablation leak test (`bench/cache_leak_ablation.py`)
A canary is planted in the victim's cached answer; one bug is injected at a time; layers are then disabled progressively. "Stopped by" is inferred from which layer must be removed before the leak appears.

| injected bug | attack | layers off | leaked | stopped by | event |
|---|---|---|---|---|---|
| none (control) | cross-tenant | – | no | key/keyspace isolation | – |
| key omits tenant | cross-tenant | – | no | 4 AEAD | decrypt_failure |
| … AEAD off | cross-tenant | aead | no | 5 tenant tag | tenant_tag_mismatch |
| … AEAD+tag off | cross-tenant | aead, tag | no | 6 validate-on-read | – |
| … AEAD+tag+validate off | cross-tenant | +validate | no | 7 re-materialization | – |
| … layers 4–7 off | cross-tenant | all four | **YES** | none (expected) | – |
| scope omitted from key | cross-user | – | no | 6 validate-on-read | – |
| … validate off | cross-user | validate | no | 7 re-materialization | – |
| … validate+remat off | cross-user | both | **YES** | none (expected) | – |
| shared semantic index, filter dropped | cross-tenant | – | no | 4–7 | decrypt_failure |
| ACL revoked, events lost | stale ACL | – | no | 6 validate-on-read | – |
| … validate off | stale ACL | validate | no | 7 re-materialization | – |
| … validate+remat off | stale ACL | both | **YES** | none (expected) | – |

The three "YES" rows are deliberate: they show each layer is load-bearing and independent. Correctness never depends on invalidation events being delivered.

### 6. Noisy neighbor (`bench/noisy_neighbor.py`)
Tenant A sends 2,000 req/s, tenant B sends 20 req/s, shared pool of 4 workers.

| config | B p50 | B p95 | B served | A served | A rejected |
|---|---|---|---|---|---|
| no limits | 6,180 ms | 11,746 ms | 80 | 8,000 | 0 |
| per-tenant limits | **8.2 ms** | **8.4 ms** | 80 | 299 | 7,701 (429) |

### 7. Isolation test suite
71 tests in `tests/engine/` (plus 40 legacy tests). Highlights: zero leaks in 3 tenancy modes × pre/post filtering with a deliberately tiny `ef`; revocation enforced before ACL propagation while grants wait (fail-safe) with stale hits counted; deleted/quarantined docs vanish next request; default-deny and deny-overrides ACL semantics on Postgres, Qdrant and in-memory; RLS on the new tables for a non-superuser role; unique-index scan (no cross-tenant unique constraint); key-uniqueness property test with delimiter-injection strings; ciphertext moved between tenants fails and alerts; 200 concurrent requests across two scopes cause exactly two pipeline runs; a flooding tenant cannot evict another tenant's cache entries; real Redis backend; planted prompt injection never reaches the prompt.

## What didn't work
- **Per-tenant reranker (feedback adaptation): no gain.** Pointwise logistic regression on 6 interpretable features made things *worse* (mean MRR −0.098). Adding the shared reranker's score as a feature (residual learning) improved that to −0.036. Shrinking the weights toward the shared model made lift exactly 0.0 (safe, but useless). The remaining errors on this corpus come from candidate recall and confusable names, which these features cannot fix with ~35 feedback events per tenant. The cold-start fallback and the "never worse than shared" property work; the lift does not. Real tenant-specific signal (terminology, a fine-tuned cross-encoder, an embedding adapter) is the next experiment.
- **Fixed `ef` sweep on the IVF model.** The first sweep raised `ef` but recall stayed flat because `nprobe` capped the candidate pool. Search effort has to scale both knobs.
- **Production query plan never touches HNSW** at these sizes, so the headline filtered-ANN experiment had to use isolated tables (it measures pgvector's behaviour, not the production query's).
- **A union-of-chunks sufficiency judge** let sibling documents satisfy the check (hallucination 29%); judging coverage within a single chunk fixed it (1%).
- **First synthetic corpus was too easy** (recall 1.0 everywhere); confusable subject names and typos were added so ablations separate.

## Status, limits and what is NOT done
Built and tested: tenancy router (3 modes), Postgres hybrid retriever (dense + tsvector, RLS, ACL with deny), ACL admin with outbox, Qdrant payload-multitenancy adapter (tested in local in-memory mode only), hybrid retrieval + RRF + rerank, router/rewrite/decompose, self-check loop, guardrails, PII redaction, injection detector, audit log, 7-layer cache (memory + real Redis), gateway, cost ledger, adaptation, FastAPI `/v2`, Docker/compose, CI gate.

Not done or not verified:
- **Gemini/Vertex generation and the Hugging Face cross-encoder are adapters that were never run** (no credentials/model downloads in the build environment). All generation results use the deterministic `ExtractiveLLM`.
- **Embeddings are a hashing n-gram model.** No semantic or cross-lingual retrieval was evaluated; Hindi/Gujarati support is limited to tokenization, `simple` tsvector config, injection patterns and cache exact-only fallback.
- **No BEIR or multilingual benchmark run**, and no real public corpus (a directory loader exists: `load_directory_corpus`).
- **Qdrant server, pgvector ≥ 0.8 iterative scans, and a Postgres partition-promotion benchmark were not run.** The `document_versions` / version-swap / `chunk_vectors_*` schema from the design notes is **not** implemented; this repo extends the existing schema with migration `pg0005`.
- **ACL arrays vs ACL objects, timing-leak classifier AUC, shadow validation, and semantic-threshold calibration on real paraphrase data** are designed but not run. Cache thresholds are placeholders.
- Injection/PII detectors are rule-based and were checked on a *tiny* labeled set (7 attacks, 6 benign); the numbers say little about real-world precision.
- Cost figures are a **model** with editable assumptions, not billing data. The OpenTelemetry module is a no-op unless installed; no Grafana dashboards or admin UI were built.
- The deny-permission change to the legacy `PgVectorRetriever` is covered only indirectly (the 40 legacy tests pass; the new retriever's deny behaviour is tested directly).

## Changes to legacy code
- `app/infrastructure/vectorstore/pgvector_store.py`: the ACL query now honours explicit `deny` grants (a deny was previously impossible to express). Migration `pg0005` extends `document_acl.permission` to allow `deny`.

## Run it
```bash
pip install -r requirements-tenantrag.txt
# Postgres 16 + pgvector (e.g. docker compose -f docker-compose.postgres.yml up -d), then:
export POSTGRES_DATABASE_URL=postgresql://USER:PASS@localhost:5432/tenantrag
alembic -c alembic-postgres.ini upgrade head

pytest -q tests/engine                       # 71 tests (PG tests need POSTGRES_DATABASE_URL)
python -m tenantrag.eval.gate                # CI quality gate (compares to eval/baseline.json)
python bench/tenancy_scaling.py              # tenancy models, 10/100/1000 tenants
python bench/pg_filtered_recall.py           # real pgvector filtered-ANN recall
python bench/ablations.py                    # retrieval + self-check ablations
python bench/cache_leak_ablation.py --check  # cache layer-ablation leak test
python bench/noisy_neighbor.py               # noisy-neighbor experiment
python bench/adaptation.py                   # per-tenant adaptation
uvicorn tenantrag.api.demo:app --port 8100   # demo API (prints demo bearer tokens)
docker compose -f docker-compose.tenantrag.yml up --build   # demo stack; not the upload-to-answer API
```
CI (`.github/workflows/tenantrag.yml`) runs the tests, the quality gate (fails the PR on a quality regression or any leak) and the cache leak ablation. Latency is reported but never gated (shared runners are too noisy).

## Upload-to-answer RAG workflow (PostgreSQL + Ollama)

Use this workflow to query your own documents. It runs the enterprise API
(`app.main:app`) with tenant-scoped PostgreSQL/pgvector storage and Ollama for
both embeddings and answer generation. It is separate from the synthetic,
in-memory `tenantrag.api.demo` on port 8100.

### 1. Configure and start local services (PowerShell)

```powershell
Copy-Item .env.example .env
# Edit .env: set strong POSTGRES_PASSWORD and both database URLs; keep
# DATABASE_BACKEND=postgres and use separate admin/runtime database passwords.
docker compose -f docker-compose.postgres.yml up -d
docker compose -f docker-compose.postgres.yml exec ollama ollama pull nomic-embed-text
docker compose -f docker-compose.postgres.yml exec ollama ollama pull qwen2.5:3b
```

Install the full application dependencies and apply the PostgreSQL migration
chain (not the legacy MySQL chain):

```powershell
pip install -r requirements.txt
alembic -c alembic-postgres.ini upgrade head
```

Create the restricted `tenantrag_runtime` role and grant its table/function
permissions as documented in
[`docs/tenantrag/postgres-foundation.md`](docs/tenantrag/postgres-foundation.md).
The API must use this non-superuser role; the admin URL is only for migrations
and tenant provisioning. The example file has placeholders and must be edited.

### 2. Create a tenant API key

The command reads `POSTGRES_ADMIN_DATABASE_URL` from `.env`. The plaintext key
is printed once; keep it private.

```powershell
python scripts/provision_tenant_api_key.py --tenant-id demo-tenant --tenant-name "Demo Tenant"
```

If the tenant already exists, omit `--tenant-name`.

### 3. Start the API and wait for readiness

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`GET http://127.0.0.1:8000/health` returns 503 while PostgreSQL or either
configured Ollama model is unavailable. Wait for `database: ok` and
`ai_assistant: ok`; model connection failures are logged and surfaced by the
health endpoint. Swagger UI is at `http://127.0.0.1:8000/docs`.

### 4. Upload, ask, and read back the conversation

Use `curl.exe` from PowerShell, replacing the key and providing a local UTF-8
text file:

```powershell
$tenantKey = Read-Host "Tenant API key"
curl.exe -X POST "http://127.0.0.1:8000/documents/upload" `
  -H "X-Tenant-API-Key: $tenantKey" `
  -F "file=@.\policy.txt" -F "language=en"

$session = curl.exe -s -X POST "http://127.0.0.1:8000/create-session" `
  -H "X-Tenant-API-Key: $tenantKey" -H "Content-Type: application/json" `
  -d '{"tenant_id":"demo-tenant","customer_id":"customer-1"}' | ConvertFrom-Json

$sessionToken = $session.session_api_token
$customerToken = $session.customer_token
$sessionId = $session.session_id

curl.exe -X POST "http://127.0.0.1:8000/ask?session_id=$sessionId&lang=en" `
  -H "Authorization: Bearer $sessionToken" -H "Content-Type: application/json" `
  -d '{"query":"What does the policy say?"}'

curl.exe "http://127.0.0.1:8000/chat-history?session_id=$sessionId" `
  -H "Authorization: Bearer $customerToken"
```

The answer includes retrieved source chunks. Uploads are limited to 10 MiB and
support UTF-8 `.txt` or text-based `.pdf` files; files are not written to the
local filesystem. The `/v2/ask` endpoint belongs to the separate in-memory
demo and does not read uploaded PostgreSQL documents.

## Previous Enterprise API notes (superseded)

Use the upload-to-answer workflow above instead; the notes below describe the
older setup and are retained only for history.

The enterprise API is `app.main:app`. It persists tenant sessions and chat history in PostgreSQL and exposes a tenant-authenticated document upload endpoint. This is separate from the in-memory `tenantrag.api.demo` used for a quick demo.

1. Install the enterprise dependencies with `uv pip install -r requirements.txt`. Start PostgreSQL and apply the `alembic_postgres` migrations. Configure `DATABASE_BACKEND=postgres`, `POSTGRES_DATABASE_URL` (the restricted runtime role), and `POSTGRES_ADMIN_DATABASE_URL` (migration/provisioning role); follow [`docs/tenantrag/postgres-foundation.md`](docs/tenantrag/postgres-foundation.md) for role setup.
2. Provision a tenant API key for an existing tenant with `scripts/provision_tenant_api_key.py`. Keep the printed key private; use it as `X-Tenant-API-Key`.
3. Start the API with `uv run --no-sync python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`. PostgreSQL mode also requires the configured Ollama embedding model; the legacy assistant's language model must be configured as described in the deployment docs.
4. Open `http://127.0.0.1:8000/docs`. Check `GET /health`, then call `POST /documents/upload` with the tenant key, a `.txt` or text-based `.pdf`, and an optional language tag. The upload is chunked, embedded, stored in PostgreSQL/pgvector, and made readable to authenticated users of that tenant.
5. Call `POST /create-session` with the same tenant key and `tenant_id`/`customer_id`. Use the returned `session_api_token` as a Bearer token with `GET /get-questions?session_id=...` and `POST /ask?session_id=...&lang=en`. Use the returned `customer_token` as a Bearer token with `GET /chat-history?session_id=...` to read the persisted conversation.

The upload endpoint is intentionally PostgreSQL-only. Files are limited to 10 MiB, must be UTF-8 text or text-based PDF, and are not written to the local filesystem. The demo `/v2/ask` remains in-memory and does not see uploaded documents.

## Repo layout
```
tenantrag/            engine: types, text, doc_store, corpus/, eval/, retrieval/, tenancy/, pipeline/,
                      cache/, gateway/, security/, adaptation/, observability/, api/
alembic_postgres/     migrations (pg0005 adds tsvector, policy columns, audit, feedback, usage, ACL outbox, deny)
bench/                reproducible experiments        results/   their committed outputs
tests/engine/         71 tests                         eval/      CI baseline
app/ main.py ...      legacy chatbot (kept)
```
