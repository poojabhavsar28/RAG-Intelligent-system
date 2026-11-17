"""TenantRAG: lexical leg, document policy columns, audit, feedback, usage, ACL outbox.

Revision ID: pg0005
Revises: pg0004

Adds (all tenant tables get FORCE ROW LEVEL SECURITY, same policy as pg0001):
  * document_chunks.tsv  generated tsvector + GIN index  (BM25-style lexical leg)
  * document_acl accepts permission='deny' (deny overrides every grant)
  * documents policy columns: acl_version, trust_tier, sensitivity, injection_state,
    legal_hold, deleted_at
  * acl_events   outbox consumed by cache invalidation / ACL propagation workers
  * audit_log    who retrieved which documents (query hash only, never query text)
  * feedback     thumbs / clicked-citation signals for per-tenant adaptation
  * usage_ledger per-request token + cost accounting by tenant

The 'simple' text-search configuration is used on purpose: PostgreSQL ships no
Hindi or Gujarati dictionaries, so tokenization is language-neutral.
"""
from alembic import op

revision = "pg0005"
down_revision = "pg0004"
branch_labels = None
depends_on = None

NEW_TABLES = ("acl_events", "audit_log", "feedback", "usage_ledger")


def upgrade() -> None:
    op.execute(
        "ALTER TABLE document_chunks ADD COLUMN tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED"
    )
    op.execute("CREATE INDEX idx_document_chunks_tsv ON document_chunks USING gin (tsv)")

    op.execute("""
        ALTER TABLE documents
            ADD COLUMN acl_version INTEGER NOT NULL DEFAULT 1,
            ADD COLUMN trust_tier SMALLINT NOT NULL DEFAULT 1 CHECK (trust_tier BETWEEN 0 AND 3),
            ADD COLUMN sensitivity SMALLINT NOT NULL DEFAULT 0,
            ADD COLUMN injection_state TEXT NOT NULL DEFAULT 'clean'
                CHECK (injection_state IN ('clean', 'suspect')),
            ADD COLUMN legal_hold BOOLEAN NOT NULL DEFAULT false,
            ADD COLUMN deleted_at TIMESTAMPTZ
    """)

    # Explicit deny grants (deny always wins), needed for deny-overrides ACL semantics.
    op.execute("ALTER TABLE document_acl DROP CONSTRAINT IF EXISTS document_acl_permission_check")
    op.execute(
        "ALTER TABLE document_acl ADD CONSTRAINT document_acl_permission_check "
        "CHECK (permission IN ('read', 'write', 'admin', 'deny'))"
    )

    op.execute("""
        CREATE TABLE acl_events (
            event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            document_id UUID NOT NULL,
            acl_version INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            applied_at TIMESTAMPTZ
        )
    """)
    op.execute("CREATE INDEX idx_acl_events_pending ON acl_events(tenant_id, event_id) WHERE applied_at IS NULL")

    op.execute("""
        CREATE TABLE audit_log (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            user_id TEXT NOT NULL,
            query_hash TEXT NOT NULL,
            document_ids TEXT[] NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX idx_audit_log_tenant_time ON audit_log(tenant_id, created_at)")

    op.execute("""
        CREATE TABLE feedback (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            user_id TEXT NOT NULL,
            query TEXT NOT NULL,
            document_id TEXT NOT NULL,
            signal SMALLINT NOT NULL CHECK (signal IN (-1, 0, 1)),
            clicked BOOLEAN NOT NULL DEFAULT false,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX idx_feedback_tenant_time ON feedback(tenant_id, created_at)")

    op.execute("""
        CREATE TABLE usage_ledger (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            request_id TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            embed_calls INTEGER NOT NULL DEFAULT 0,
            cache TEXT NOT NULL DEFAULT 'miss' CHECK (cache IN ('miss', 'hit_exact', 'hit_semantic', 'bypass')),
            usd NUMERIC(12, 6) NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX idx_usage_ledger_tenant_time ON usage_ledger(tenant_id, created_at)")

    for table in NEW_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation_{table} ON {table} "
            "USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')) "
            "WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''))"
        )


def downgrade() -> None:
    op.execute("DELETE FROM document_acl WHERE permission = 'deny'")
    op.execute("ALTER TABLE document_acl DROP CONSTRAINT IF EXISTS document_acl_permission_check")
    op.execute(
        "ALTER TABLE document_acl ADD CONSTRAINT document_acl_permission_check "
        "CHECK (permission IN ('read', 'write', 'admin'))"
    )
    for table in reversed(NEW_TABLES):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    op.execute("""
        ALTER TABLE documents
            DROP COLUMN IF EXISTS deleted_at, DROP COLUMN IF EXISTS legal_hold,
            DROP COLUMN IF EXISTS injection_state, DROP COLUMN IF EXISTS sensitivity,
            DROP COLUMN IF EXISTS trust_tier, DROP COLUMN IF EXISTS acl_version
    """)
    op.execute("DROP INDEX IF EXISTS idx_document_chunks_tsv")
    op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS tsv")
