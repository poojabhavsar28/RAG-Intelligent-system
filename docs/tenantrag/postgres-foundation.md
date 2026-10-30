# TenantRAG PostgreSQL foundation

## Scope of this increment

This branch now contains a **separate PostgreSQL migration history** and a
transaction helper for setting PostgreSQL row-level-security (RLS) context.
The existing MySQL runtime and MySQL Alembic history are intentionally left
untouched while the PostgreSQL repositories and service wiring are built.

## Start PostgreSQL

Set a strong local password before starting the service:

```bash
# PowerShell
$env:POSTGRES_PASSWORD = "replace-with-a-local-secret"
docker compose -f docker-compose.postgres.yml up -d
```

Set the application migration URL (URL-encode special characters in passwords):

```bash
# PowerShell
$env:POSTGRES_DATABASE_URL = "postgresql://tenantrag_app:replace-with-a-local-secret@localhost:5432/tenantrag"
```

Install dependencies and apply the PostgreSQL schema:

```bash
pip install -r requirements.txt
alembic -c alembic-postgres.ini upgrade head
```

## Tenant isolation contract

1. Authenticate the caller using the existing session/token flow.
2. Resolve the tenant UUID from the authenticated server-side session.
3. Open a transaction with `tenant_transaction(engine, trusted_tenant_id)`.
4. Perform all tenant-owned reads and writes on the yielded connection.
5. Never accept `app.tenant_id` from request payloads or caller-controlled headers.

The transaction helper uses PostgreSQL `SET LOCAL` semantics through
`set_config(..., true)`, so the setting is scoped to the transaction and
does not leak across pooled connections.

The application database role must not be a superuser and must not have
`BYPASSRLS`. Table owners are subject to RLS because the migration enables
`FORCE ROW LEVEL SECURITY`.

## Data-plane design

- PostgreSQL is the canonical metadata, session, token, document and chunk
  store for the new data plane.
- `vector(768)` matches the current Nomic Embed Text baseline. Changing the
  embedding model/dimension requires an explicit migration and re-embedding.
- Document and chunk tables use 16-way hash partitioning on `tenant_id`.
  This bounds partition count while distributing tenants; it is not one
  physical partition per tenant.
- RLS enforces tenant boundary. Document ACL checks remain an additional
  authorization filter and must be applied by the retrieval query.
- HNSW cosine index is the initial approximate-nearest-neighbor baseline.

## Current migration boundary

The original `alembic.ini` and `alembic/` directory are MySQL-specific.
Do not run the old MySQL migration chain against PostgreSQL. Use
`alembic-postgres.ini` for the new data plane.

## Not yet complete

The current API repositories and `AIAssistant` still use MySQL/Chroma.
They have not yet been switched to PostgreSQL. The next implementation
increment must add PostgreSQL repositories, tenant-scoped hybrid retrieval,
and service wiring, then port sessions, tokens, chat history and FAQ access.
No migration or benchmark has been executed against a live PostgreSQL server
in this commit, so runtime success and performance numbers are not claimed.
