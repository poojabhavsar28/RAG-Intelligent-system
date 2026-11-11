# Runtime Validation Report

**Generated**: 2026-11-08
**Target**: `TenantRAG-full-repo` enterprise API

## Summary

| Step | Status | Exit Code | Details |
|---|---|---:|---|
| Startup | BLOCKED | 1 | `uv run --no-sync python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`; configuration rejected because PostgreSQL and MySQL settings are not configured. |
| API workflow tests | PASS | 0 | Upload, session creation, FAQ lookup, ask, chat-history readback, and PostgreSQL health route exercised through FastAPI `TestClient` with test doubles. |
| Full tests | PASS | 0 | `uv run --no-sync python -m pytest -o addopts= -q tests`; 104 passed, 13 skipped. |
| Live PostgreSQL workflow | UNVERIFIED | N/A | Docker unavailable, local PostgreSQL port 5432 closed, and no PostgreSQL URLs configured. |

**Overall**: PARTIAL — route behavior passed, but real PostgreSQL persistence, Ollama embeddings, and a live Uvicorn deployment remain unverified.

## Environment

- Docker: unavailable (not installed).
- Node.js: available (`v24.17.0`); browser E2E is not applicable to this REST API.
- PostgreSQL client: unavailable; `127.0.0.1:5432` is not accepting connections.
- HTTP server: neither port 8000 nor 8100 was responding before validation.
- PostgreSQL URL and tenant API key: not configured in the validation environment.

## API workflow evidence

The HTTP-level workflow test performs:

1. `GET /health` against a fake PostgreSQL connection.
2. `POST /documents/upload` with a tenant API key and a text file.
3. `POST /create-session`.
4. `GET /get-questions` with the returned session identity.
5. `POST /ask` with the session bearer token.
6. `GET /chat-history` with the customer bearer token and verifies the recorded exchange.

Test file: `tests/test_enterprise_api_workflow.py`.

The workflow uses test doubles for PostgreSQL, embedding, and application services. It verifies FastAPI route/auth/orchestration behavior but does **not** prove durable database writes or real retrieval from pgvector. PostgreSQL-backed tests were skipped because `POSTGRES_DATABASE_URL` was not configured.

## Issues and follow-up

- Enterprise startup requires local PostgreSQL settings (`DATABASE_BACKEND=postgres`, `POSTGRES_DATABASE_URL`, and `POSTGRES_ADMIN_DATABASE_URL` for migrations/provisioning).
- A provisioned tenant API key and running Ollama embedding/model services are needed for live upload-and-ask validation.
- PostgreSQL health routing was adjusted to ping the configured PostgreSQL engine; mocked health-route coverage passes.
- Remaining test warnings are non-fatal: missing `pytest-asyncio` configuration support, Starlette/httpx deprecation, local Qdrant payload-index limitation, and PyPDF2 deprecation.
