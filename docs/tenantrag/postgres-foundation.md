# TenantRAG PostgreSQL foundation

## Scope of this increment

This branch now contains a **separate PostgreSQL migration history**, a
transaction helper for setting PostgreSQL row-level-security (RLS) context,
a hash-only authentication bootstrap index, and initial PostgreSQL repository
adapters. The existing MySQL runtime and MySQL Alembic history are intentionally
left untouched while authentication and service wiring are completed.

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

The PostgreSQL repository adapters now cover tenant lookup, session creation
and lookup, customer-token creation and validation, and chat-history reads
and writes. They use the RLS transaction helper for tenant-owned tables and
the hash-only bootstrap index for pre-RLS token resolution.

The FastAPI service graph and `AIAssistant` are **not yet switched**: the
active runtime still uses MySQL/Chroma. Do not treat the new repositories as
a completed database cutover.

Before wiring session creation to PostgreSQL, replace the current
caller-supplied `tenant_id` request field as the source of authority with a
trusted tenant resolver (for example, a tenant API credential or verified
host/domain mapping). Otherwise a caller could request session creation for
a tenant by guessing its identifier. Session and customer-token creation
also still need a single atomic transaction at the service boundary.

GitHub Actions has successfully applied both PostgreSQL migrations,
compiled the PostgreSQL modules, passed tenant-ID and pgvector contract
tests, and exercised tenant RLS using a non-superuser runtime role. This
validates the schema and isolation foundation; it is not yet an end-to-end
FastAPI integration test or a production performance benchmark.
