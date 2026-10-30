# TenantRAG PostgreSQL foundation

## Scope of this increment

This branch now contains a **separate PostgreSQL migration history**, a
transaction helper for setting PostgreSQL row-level-security (RLS) context,
a hash-only authentication bootstrap index, PostgreSQL repository adapters,
an ACL-aware pgvector retriever, migration utilities, and an opt-in PostgreSQL
application backend. MySQL remains the default backend for backward
compatibility; the PostgreSQL backend is selected explicitly after data
migration and tenant API-key provisioning.

## Start PostgreSQL

Set a strong local password before starting the service:

```bash
# PowerShell
$env:POSTGRES_PASSWORD = "replace-with-a-local-secret"
docker compose -f docker-compose.postgres.yml up -d
```

Use separate credentials for schema changes and application traffic.
The compose bootstrap user (`tenantrag_app`) is the privileged migration
identity; the application must use a non-superuser role without `BYPASSRLS`.
URL-encode special characters in passwords.

```bash
# PowerShell: privileged migration/admin URL
$env:POSTGRES_ADMIN_DATABASE_URL = "postgresql://tenantrag_app:ADMIN_PASSWORD@localhost:5432/tenantrag"
```

Install dependencies and apply the PostgreSQL schema:

```bash
pip install -r requirements.txt
alembic -c alembic-postgres.ini upgrade head
```

Create the restricted runtime role using the admin connection (choose a
strong password and keep it separate from the migration password):

```sql
CREATE ROLE tenantrag_runtime LOGIN PASSWORD 'RUNTIME_PASSWORD'
  NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
GRANT CONNECT ON DATABASE tenantrag TO tenantrag_runtime;
GRANT USAGE ON SCHEMA public TO tenantrag_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO tenantrag_runtime;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO tenantrag_runtime;
ALTER DEFAULT PRIVILEGES FOR ROLE tenantrag_app IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO tenantrag_runtime;
ALTER DEFAULT PRIVILEGES FOR ROLE tenantrag_app IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO tenantrag_runtime;
```

Then configure the runtime connection and select the PostgreSQL backend:

```bash
# PowerShell
$env:POSTGRES_DATABASE_URL = "postgresql://tenantrag_runtime:RUNTIME_PASSWORD@localhost:5432/tenantrag"
$env:DATABASE_BACKEND = "postgres"
```

Apply MySQL metadata and Chroma imports below before switching production
traffic. The app's PostgreSQL path uses Ollama for Nomic embeddings and the
existing local Llama model for answer generation.

## Tenant isolation contract

1. Authenticate the caller using the existing session/token flow.
2. Resolve the tenant ID string from the authenticated server-side session.
3. For bearer-token authentication, resolve the tenant through
   `auth_token_lookup` using the token hash and expected token type.
4. Open a transaction with `tenant_transaction(engine, trusted_tenant_id)`.
5. Perform all tenant-owned reads and writes on the yielded connection.
6. Never accept `app.tenant_id` from request payloads or caller-controlled headers.

Tenant IDs are stored as `TEXT` throughout the PostgreSQL schema. Existing IDs are preserved byte-for-byte as strings; they are not converted to UUIDs or regenerated. Internal resource IDs such as session, document and chunk IDs may still use UUIDs. The transaction helper uses PostgreSQL `SET LOCAL` semantics through
`set_config(..., true)`, so the setting is scoped to the transaction and
does not leak across pooled connections.

The `auth_token_lookup` table is the deliberate exception to tenant RLS:
authentication needs a minimal lookup before the tenant is known. It stores
only token hashes, token type, tenant ID, subject ID, session ID (when
applicable), and expiry/revocation timestamps. Restrict access to this table
to the authentication/repository layer; never expose it through a general
query endpoint.

The application database role must not be a superuser and must not have
`BYPASSRLS`. Table owners are subject to RLS because the migration enables
`FORCE ROW LEVEL SECURITY`.

## Legacy MySQL metadata import

After applying the PostgreSQL migrations, the initial restartable importer is
available at `scripts/migrate_mysql_to_postgres.py`. It copies tenant rows,
sessions, customer tokens, chat history, and FAQ templates in batches. It
preserves tenant IDs exactly and imports legacy chat/FAQ row IDs so reruns do
not duplicate those rows.

Configure the existing MySQL connection variables
(`MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DB`)
and `POSTGRES_ADMIN_DATABASE_URL`, then run:

```bash
python scripts/migrate_mysql_to_postgres.py
```

The PostgreSQL URL must use a dedicated migration administrator with
`BYPASSRLS`; do not use the restricted application role for bulk import.
Back up both databases first. The importer intentionally stops if a legacy
session ID is not a UUID rather than generating a replacement.

The vector collections have a separate importer:
`scripts/migrate_chroma_to_pgvector.py`. Run it after the metadata importer
with the same MySQL variables, `CHROMA_DB_BASE_DIR`, and
`POSTGRES_ADMIN_DATABASE_URL` configured:

```bash
python scripts/migrate_chroma_to_pgvector.py
```

It reads each tenant/language path from MySQL `chroma_metadata`, copies
existing embeddings directly into `document_chunks`, and creates stable
document/chunk UUIDs from legacy source identifiers. It fails closed if a
vector is not 768-dimensional or a collection path escapes
`CHROMA_DB_BASE_DIR`. Imported legacy chunks receive a tenant-local
`tenant_member` read ACL to preserve the old tenant-isolated access model;
the application must include `tenant_member` in the authenticated
`RetrievalContext.role_ids` when serving those chunks.

Neither importer deletes or modifies source MySQL rows or physical Chroma
directories. Keep both source systems intact until row counts, retrieval
quality, ACL behavior, and application cutover have been verified.

## Data-plane design

- PostgreSQL is the canonical metadata, session, token, document and chunk
  store for the new data plane.
- `vector(768)` matches the current Nomic Embed Text baseline. Changing the
  embedding model/dimension requires an explicit migration and re-embedding.
- Document and chunk tables use 16-way hash partitioning on `tenant_id`.
  This bounds partition count while distributing tenants; it is not one
  physical partition per tenant.
- RLS enforces tenant boundary. The initial `PgVectorRetriever` also filters
  active documents by language and applies user, group, and role ACL checks
  before returning chunks. ACL enforcement is on by default.
- HNSW cosine index is the initial approximate-nearest-neighbor baseline.
- The current retriever accepts 768-dimensional query embeddings only; a
  model/dimension change must be accompanied by a schema migration and
  re-embedding rather than mixing vector spaces.

## Current migration boundary

The original `alembic.ini` and `alembic/` directory are MySQL-specific.
Do not run the old MySQL migration chain against PostgreSQL. Use
`alembic-postgres.ini` for the new data plane.

## Current implementation boundary

The optional PostgreSQL application path is now wired in `app.main` and is
selected with `DATABASE_BACKEND=postgres`. It uses PostgreSQL repositories
for sessions, customer tokens, chat history, tenants, and FAQ templates;
session plus customer-token creation is atomic. The session-creation endpoint
requires `X-Tenant-API-Key` and resolves the trusted tenant from the
hash-only bootstrap table. Provision keys with
`scripts/provision_tenant_api_key.py` after the tenant has been imported.

The PostgreSQL RAG path uses `PgVectorRetriever` with tenant RLS, language,
active-document, and user/group/role ACL filters. The current assistant adds
the legacy `tenant_member` role to preserve access to imported Chroma
collections. The MySQL backend remains the default unless
`DATABASE_BACKEND=postgres` is explicitly set.

The relational importer and Chroma-to-pgvector importer are committed, but
neither has been executed against your actual databases or Chroma directories.
Do not switch production traffic until you have backed up the sources, run
both imports, reconciled row/chunk counts, provisioned tenant API keys, and
tested real API requests. Chroma collection paths and embedding dimensions
are validated by the importer; no source data is deleted.

GitHub Actions has successfully applied the PostgreSQL migrations, compiled
the application and migration modules, passed tenant-ID, migration-helper,
and pgvector contract tests, and exercised RLS plus ACL-aware vector retrieval
using a non-superuser runtime role. These are CI integration checks, not a
production load benchmark or proof of successful migration of your data.
