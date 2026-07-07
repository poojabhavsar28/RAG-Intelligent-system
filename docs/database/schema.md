# Database Schema

## Entity-relationship diagram

```mermaid
erDiagram
    TENANTS ||--o{ SESSIONS : "has"
    TENANTS ||--o{ CUSTOMER_TOKENS : "has"
    TENANTS ||--o{ CHROMA_METADATA : "has"
    TENANTS ||--o{ QA_TEMPLATES : "has"
    TENANTS ||--o{ CHAT_HISTORY : "has"
    SESSIONS ||--o{ CHAT_HISTORY : "produces"

    TENANTS {
        varchar tenant_id PK
        varchar name
        varchar domain
        timestamp created_at
    }
    SESSIONS {
        varchar session_id PK
        varchar tenant_id FK
        varchar customer_id
        varchar api_token_hash
        varchar language
        timestamp created_at
        timestamp expires_at
    }
    CHAT_HISTORY {
        int id PK
        varchar session_id FK
        varchar tenant_id FK
        varchar role
        text message
        varchar sender
        timestamp timestamp
    }
    CUSTOMER_TOKENS {
        varchar token_hash PK
        varchar tenant_id FK
        varchar customer_id
        timestamp created_at
        timestamp expires_at
    }
    CHROMA_METADATA {
        varchar tenant_id PK_FK
        varchar lang PK
        varchar db_path
        timestamp created_at
    }
    QA_TEMPLATES {
        int id PK
        varchar tenant_id FK
        varchar lang
        text question
        text answer
        timestamp created_at
        timestamp updated_at
    }
```

## Tables

**`tenants`** — one row per client organization using the chatbot (e.g. a bank, an insurer). This is the canonical source of truth for `tenant_id`; every other table's `tenant_id` column is a foreign key into this table. Added in migration `0002` — before that, `tenant_id` was a free-text string repeated across five tables with nothing preventing a typo from silently creating a "new" tenant that no code or admin screen knew about.

**`sessions`** — one row per chat session a customer starts. **One tenant has many sessions** (a bank has many customers, each customer's visit starts a session) — modeled as `sessions.tenant_id → tenants.tenant_id`.

**`chat_history`** — one row per message (both the user's question and the assistant's answer are separate rows) within a session. **One session has many chat_history rows**, modeled as `chat_history.session_id → sessions.session_id`. `tenant_id` is *also* stored directly on this table even though it's derivable via a join through `sessions` — this is a deliberate denormalization, not an oversight: tenant-scoped chat queries (e.g. "all conversations for this bank this month") are common enough to be worth avoiding a join for, at the cost of the column needing to stay consistent with its parent session's tenant (enforced at the application layer in `ChatRepository.append_exchange`, since both values come from the same validated session lookup in `ChatService`).

**`customer_tokens`** — one row per issued customer-level bearer token (used to authorize `/chat-history` access, distinct from the per-session token used for `/ask`). **One tenant has many customer tokens.**

**`chroma_metadata`** — one row per (tenant, language) vector store collection, recording where that collection's data lives on disk. **One tenant has many chroma_metadata rows** (one per language it operates in — English and Hindi today).

**`qa_templates`** — one row per predefined question/answer pair surfaced by `/get-questions`, scoped per tenant and language. **One tenant has many qa_templates.**

## Foreign keys and why each `ON DELETE` behavior was chosen

| Relationship | Behavior | Why |
|---|---|---|
| `sessions.tenant_id → tenants` | `RESTRICT` | A tenant with active sessions shouldn't be deletable without resolving those sessions first — cascade-deleting a tenant's entire session history as a side effect is too destructive to happen implicitly. |
| `customer_tokens.tenant_id → tenants` | `RESTRICT` | Same reasoning as sessions. |
| `chroma_metadata.tenant_id → tenants` | `RESTRICT` | This row points at a real directory on disk. Cascade-deleting the row would silently orphan that directory instead of triggering an explicit cleanup step that actually removes the files. |
| `qa_templates.tenant_id → tenants` | `CASCADE` | Tenant-owned configuration content with no meaning outside that tenant and no external side effects (no files, no other table references it) — safe to remove along with the tenant. |
| `chat_history.tenant_id → tenants` | `RESTRICT` | Consistent with sessions/tokens. |
| `chat_history.session_id → sessions` | `CASCADE` | Chat history has no independent meaning once its session is gone. The pre-migration schema had this FK with no explicit `ON DELETE`, which defaults to `RESTRICT` in InnoDB — meaning a session could never actually be deleted while it had any history, with no clear error explaining why. Fixed in migration `0002`. |

## What was considered and deliberately not applied

- **Soft delete** — not applied anywhere. No table in this schema has a stated business requirement to recover or audit deleted rows after the fact. Hard delete (backed by the `RESTRICT`/`CASCADE` behavior above) is simpler and avoids the classic "soft-deleted row still shows up in some query somewhere" bug class. If a real audit requirement emerges later (e.g. regulatory retention for a banking tenant's chat history), that's a deliberate, scoped addition at that point — not a default to apply everywhere up front.
- **Optimistic locking (version columns)** — not applied. None of these tables have a realistic concurrent-write scenario: sessions and tokens are written once and read many times; chat history is append-only; `qa_templates` is low-frequency admin content. Adding version columns here would be solving a problem that doesn't exist yet.
- **CQRS / event sourcing** — not applied. This system has no read/write split need — read and write volumes and patterns are symmetric enough that a single schema serving both is the right level of complexity.
- **Additional normalization of `chat_history.tenant_id`** — see the denormalization note above; this one *was* considered and kept for a stated performance reason, not left in out of inertia.

## Migrations

Schema changes are managed with Alembic (`alembic/`), not the old hand-rolled `create_tables_if_not_exists()` bootstrap approach (removed in this phase). Run migrations explicitly before starting the app:

```bash
alembic upgrade head
uvicorn app.main:app
```

To create a new migration after changing the schema:
```bash
alembic revision -m "describe the change"
```
(Migrations in this project are hand-written raw SQL via `op.execute()`, not autogenerated from ORM models — the app itself doesn't use an ORM, so there's no model metadata for Alembic to diff against. `alembic revision --autogenerate` will not produce anything useful here.)

## Backup strategy

Not automated yet (no deployment target with managed backups is configured — see the upcoming DevOps phase for hosting decisions). For local development, a manual dump is sufficient:
```bash
mysqldump -h 127.0.0.1 -u raguser -p rag_chatbot > backup_$(date +%Y%m%d).sql
```
Once a managed hosting provider is chosen (Phase 7), its automatic backup feature should be enabled and documented here rather than relying on manual dumps.
