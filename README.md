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
72 tests in `tests/engine/` (plus 40 legacy tests). Highlights: zero leaks in 3 tenancy modes × pre/post filtering with a deliberately tiny `ef`; revocation enforced before ACL propagation while grants wait (fail-safe) with stale hits counted; deleted/quarantined docs vanish next request; default-deny and deny-overrides ACL semantics on Postgres, Qdrant and in-memory; RLS on the new tables for a non-superuser role; unique-index scan (no cross-tenant unique constraint); key-uniqueness property test with delimiter-injection strings; ciphertext moved between tenants fails and alerts; 200 concurrent requests across two scopes cause exactly two pipeline runs; a flooding tenant cannot evict another tenant's cache entries; real Redis backend; planted prompt injection never reaches the prompt.

## What didn't work
- **Per-tenant reranker (feedback adaptation): no gain.** Pointwise logistic regression on 6 interpretable features made things *worse* (mean MRR −0.098). Adding the shared reranker's score as a feature (residual learning) improved that to −0.036. Shrinking the weights toward the shared model made lift exactly 0.0 (safe, but useless). The remaining errors on this corpus come from candidate recall and confusable names, which these features cannot fix with ~35 feedback events per tenant. The cold-start fallback and the "never worse than shared" property work; the lift does not. Real tenant-specific signal (terminology, a fine-tuned cross-encoder, an embedding adapter) is the next experiment.
- **Fixed `ef` sweep on the IVF model.** The first sweep raised `ef` but recall stayed flat because `nprobe` capped the candidate pool. Search effort has to scale both knobs.
- **Production query plan never touches HNSW** at these sizes, so the headline filtered-ANN experiment had to use isolated tables (it measures pgvector's behaviour, not the production query's).
- **A union-of-chunks sufficiency judge** let sibling documents satisfy the check (hallucination 29%); judging coverage within a single chunk fixed it (1%).
- **First synthetic corpus was too easy** (recall 1.0 everywhere); confusable subject names and typos were added so ablations separate.

## Status, limits and what is NOT done
Built and tested: tenancy router (3 modes), Postgres hybrid retriever (dense + tsvector, RLS, ACL with deny), ACL admin with outbox, Qdrant payload-multitenancy adapter (tested in local in-memory mode only), hybrid retrieval + RRF + rerank, router/rewrite/decompose, self-check loop, guardrails, PII redaction, injection detector, audit log, 7-layer cache (memory + real Redis), gateway, cost ledger, adaptation, FastAPI `/v2`, Docker/compose, CI gate.

Not done or not verified:
- **Ollama is the supported local LLM and embedding provider.** The default runtime uses `llama3.2:3b` for generation and `nomic-embed-text` for embeddings through Ollama's local HTTP API. No Gemini, Vertex AI, cloud credentials, or paid LLM API are required. The deterministic `ExtractiveLLM` and `HashingEmbedder` remain explicit test/benchmark options, not production defaults. The Hugging Face cross-encoder remains optional and must be validated separately.
- **The local default is Ollama `nomic-embed-text`; its semantic retrieval quality has not yet been benchmarked on a representative corpus.** The hashing n-gram embedder remains for deterministic synthetic tests and ablations, not as the default runtime. No cross-lingual retrieval quality is claimed; Hindi/Gujarati support is limited to tokenization, `simple` tsvector config, injection patterns and cache exact-only fallback.
- **No BEIR or multilingual benchmark run**, and no real public corpus (a directory loader exists: `load_directory_corpus`).
- **Qdrant server, pgvector ≥ 0.8 iterative scans, and a Postgres partition-promotion benchmark were not run.** The `document_versions` / version-swap / `chunk_vectors_*` schema from the design notes is **not** implemented; this repo extends the existing schema with migration `pg0005`.
- **ACL arrays vs ACL objects, timing-leak classifier AUC, shadow validation, and semantic-threshold calibration on real paraphrase data** are designed but not run. Cache thresholds are placeholders.
- Injection/PII detectors are rule-based and were checked on a *tiny* labeled set (7 attacks, 6 benign); the numbers say little about real-world precision.
- Cost figures are a **model** with editable assumptions, not billing data. The OpenTelemetry module is a no-op unless installed; no Grafana dashboards or admin UI were built.

## Changes to legacy code
- `app/infrastructure/vectorstore/pgvector_store.py`: the ACL query now honours explicit `deny` grants (a deny was previously impossible to express); covered by `test_legacy_pgvector_retriever_honours_deny_grants`. Migration `pg0005` extends `document_acl.permission` to allow `deny`.

## Run it
```bash
pip install -r requirements-tenantrag.txt
# Postgres 16 + pgvector (e.g. docker compose -f docker-compose.postgres.yml up -d), then:
export POSTGRES_DATABASE_URL=postgresql://USER:PASS@localhost:5432/tenantrag
alembic -c alembic-postgres.ini upgrade head

pytest -q tests/engine                       # 72 tests (PG tests need POSTGRES_DATABASE_URL)
python -m tenantrag.eval.gate                # CI quality gate (compares to eval/baseline.json)
python bench/tenancy_scaling.py              # tenancy models, 10/100/1000 tenants
python bench/pg_filtered_recall.py           # real pgvector filtered-ANN recall
python bench/ablations.py                    # retrieval + self-check ablations
python bench/cache_leak_ablation.py --check  # cache layer-ablation leak test
python bench/noisy_neighbor.py               # noisy-neighbor experiment
python bench/adaptation.py                   # per-tenant adaptation
ollama pull llama3.2:3b && ollama pull nomic-embed-text  # local model setup
# Set TENANTRAG_LLM_PROVIDER=ollama and TENANTRAG_EMBEDDING_PROVIDER=ollama (defaults)
uvicorn tenantrag.api.demo:app --port 8100   # demo API (prints demo bearer tokens)
# Start Ollama on the host first, then pull both models (the API container uses host.docker.internal)
ollama serve
ollama pull llama3.2:3b
ollama pull nomic-embed-text
docker compose -f docker-compose.tenantrag.yml up --build
```
CI (`.github/workflows/tenantrag.yml`) runs the tests, the quality gate (fails the PR on a quality regression or any leak) and the cache leak ablation. Latency is reported but never gated (shared runners are too noisy).

## Repo layout
```
tenantrag/            engine: types, text, doc_store, corpus/, eval/, retrieval/, tenancy/, pipeline/,
                      cache/, gateway/, security/, adaptation/, observability/, api/
alembic_postgres/     migrations (pg0005 adds tsvector, policy columns, audit, feedback, usage, ACL outbox, deny)
bench/                reproducible experiments        results/   their committed outputs
tests/engine/         72 tests                         eval/      CI baseline
app/ main.py ...      legacy chatbot (kept)
```

## Production runtime (PostgreSQL + Redis + Ollama)

The production entry point is **`tenantrag.api.production:app`**. It is separate from
`tenantrag.api.demo:app`; Docker runs the production entry point and never creates a
synthetic corpus or prints demo bearer tokens.

### Runtime services

- PostgreSQL 16 + pgvector is the durable document, chunk, ACL, audit, feedback and usage store.
- The API connects using a dedicated `tenantrag_runtime_login` role that is not a superuser
  and cannot bypass row-level security. Alembic runs separately with the admin URL.
- Redis provides atomic shared rate limits, UTC-day request quotas and in-flight limits.
- Ollama provides local generation (`llama3.2:3b`) and embeddings (`nomic-embed-text`).
- The production API deliberately starts with answer caching disabled; the existing encrypted
  in-memory cache is a testable prototype, not a distributed production cache.

### Start locally with Docker Desktop

1. Copy `.env.example` to `.env`. Set three different strong, URL-safe random values
   for `POSTGRES_PASSWORD`, `POSTGRES_RUNTIME_PASSWORD` and `REDIS_PASSWORD`
   (hex strings from `openssl rand -hex 32` work). Never commit `.env`.
2. Start Ollama on the Windows host and download the models:

   ```powershell
   ollama serve
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

3. Start the database, migrations, Redis and API:

   ```powershell
   docker compose -f docker-compose.tenantrag.yml up --build -d
   docker compose -f docker-compose.tenantrag.yml ps
   ```

4. Wait for `http://localhost:8100/healthz` to return `status=ready`. Readiness checks
   PostgreSQL, Redis and the exact Ollama model tags. The API will return 503 until all are ready.

### Create a tenant and API key

Tenant provisioning is an administrative operation. Run from the repository root after
installing `requirements-tenantrag.txt`. In PowerShell, set the admin connection string
to the mapped local database port (do not paste it into source control):

```powershell
$env:POSTGRES_ADMIN_DATABASE_URL = "postgresql://tenantrag_admin:<ADMIN_PASSWORD>@localhost:5432/tenantrag"
python scripts/provision_tenant.py --tenant-id acme --name "Acme"
```

The command prints a high-entropy API key once. Store it in a secret manager. The API stores
only its SHA-256 hash. Do not put the plaintext key in source code, logs or Git.


If you are upgrading an already-initialized PostgreSQL volume from an earlier Compose
configuration, the init script will not rerun automatically. After migration `pg0006`,
create the runtime login once from the admin connection (use a strong password):

```sql
CREATE ROLE tenantrag_runtime_login LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD '<RUNTIME_PASSWORD>';
GRANT tenantrag_runtime TO tenantrag_runtime_login;
```

### Optional: user and group identities

For user-level access, create user-bound API keys with stable identity-provider subject IDs
(do not use email addresses as the subject). Then create groups and attach users:

```powershell
python scripts/provision_user.py --tenant-id acme --external-subject "idp-subject-001"
python scripts/provision_group.py --tenant-id acme --name "HR" --external-subject "idp-subject-001" --external-subject "idp-subject-002"
```

Both scripts require the admin connection string and write only token hashes. Keep the
printed user key in a secret manager. Group IDs are tenant-scoped and should be obtained
from the provisioning result.

### Ingest and ask

Use the key in the `X-Tenant-API-Key` header. Tenant-wide documents use the default
`visibility: "tenant"`. A user-bound key may instead submit `visibility: "private"`
or `visibility: "groups"` with group IDs the user belongs to. Example PowerShell requests:

```powershell
$headers = @{ "X-Tenant-API-Key" = "<TENANT_API_KEY>" }
$document = @{
  external_id = "employee-handbook-v1"
  title = "Employee handbook"
  content = "The annual leave allowance is 20 days."
  language = "en"
  trust_tier = 1
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:8100/v1/documents" -Headers $headers -ContentType "application/json" -Body $document

$question = @{ query = "How many annual leave days are provided?" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://localhost:8100/v1/ask" -Headers $headers -ContentType "application/json" -Body $question
```

Available endpoints: `GET /livez`, `GET /healthz`, `POST /v1/documents`,
`DELETE /v1/documents/{external_id}`, `POST /v1/ask`, `POST /v1/feedback`,
and `GET /v1/usage`. Text ingestion is supported; file parsing/OCR is not part of this API yet.

**Authorization:** tenant service keys access documents granted to the `tenant_member` role.
For user-specific access, provision user-bound keys and group membership using the admin-only
scripts below. Private documents are visible only to their owner; group documents are visible
to the owner and members of the selected tenant groups. Every SQL retrieval applies tenant
filters and authoritative ACL checks. The tenant service key cannot read private/group-only docs.

### Production deployment notes

This Compose file is a reproducible single-host deployment baseline, not a claim of
high availability. Before public internet exposure, configure TLS/reverse proxy, backups and
restore drills, secret management, network isolation, resource limits, monitoring/alerting,
and an external identity integration. Redis-backed rate limits are shared across API replicas;
the current service is intentionally single worker per container. Load-test and tune pool,
quota and model settings for the target hardware before committing an SLA.
