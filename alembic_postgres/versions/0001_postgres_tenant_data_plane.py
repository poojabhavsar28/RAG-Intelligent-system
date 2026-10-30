"""Create PostgreSQL tenant data plane.

Revision ID: pg0001
Revises:
"""
from alembic import op

revision = "pg0001"
down_revision = None
branch_labels = ("postgres",)
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.execute("""
        CREATE TABLE tenants (
            tenant_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active', 'suspended', 'deleting')),
            plan TEXT NOT NULL DEFAULT 'free',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE app_users (
            user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            external_subject TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (tenant_id, external_subject),
            UNIQUE (tenant_id, user_id)
        )
    """)

    op.execute("""
        CREATE TABLE sessions (
            session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            user_id UUID,
            customer_id TEXT NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            language TEXT NOT NULL DEFAULT 'en',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at TIMESTAMPTZ,
            FOREIGN KEY (tenant_id, user_id)
                REFERENCES app_users(tenant_id, user_id),
            UNIQUE (tenant_id, session_id)
        )
    """)
    op.execute("CREATE INDEX idx_sessions_tenant_customer ON sessions(tenant_id, customer_id)")
    op.execute("CREATE INDEX idx_sessions_expiry ON sessions(expires_at) WHERE expires_at IS NOT NULL")

    op.execute("""
        CREATE TABLE customer_tokens (
            token_hash TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            customer_id TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at TIMESTAMPTZ
        )
    """)
    op.execute("CREATE INDEX idx_customer_tokens_tenant_customer ON customer_tokens(tenant_id, customer_id)")
    op.execute("CREATE INDEX idx_customer_tokens_expiry ON customer_tokens(expires_at) WHERE expires_at IS NOT NULL")

    op.execute("""
        CREATE TABLE chat_history (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            session_id UUID NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            message TEXT NOT NULL,
            sender TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            FOREIGN KEY (tenant_id, session_id)
                REFERENCES sessions(tenant_id, session_id) ON DELETE CASCADE
        )
    """)
    op.execute("CREATE INDEX idx_chat_history_tenant_session_time ON chat_history(tenant_id, session_id, created_at)")

    op.execute("""
        CREATE TABLE qa_templates (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            language TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (tenant_id, language, question)
        )
    """)

    op.execute("""
        CREATE TABLE documents (
            tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            document_id UUID NOT NULL DEFAULT gen_random_uuid(),
            external_id TEXT,
            title TEXT NOT NULL,
            source_uri TEXT,
            language TEXT NOT NULL DEFAULT 'en',
            checksum TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('pending', 'active', 'failed', 'deleted')),
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, document_id),
            UNIQUE (tenant_id, checksum)
        ) PARTITION BY HASH (tenant_id)
    """)
    for partition in range(16):
        op.execute(
            f"CREATE TABLE documents_p{partition} PARTITION OF documents "
            f"FOR VALUES WITH (MODULUS 16, REMAINDER {partition})"
        )

    # Nomic Embed Text baseline is 768-dimensional. The dimension is explicit
    # so a model change requires a deliberate migration, not silent mixing.
    op.execute("""
        CREATE TABLE document_chunks (
            tenant_id TEXT NOT NULL,
            chunk_id UUID NOT NULL DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL,
            ordinal INTEGER NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            language TEXT NOT NULL DEFAULT 'en',
            embedding VECTOR(768),
            token_count INTEGER,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, chunk_id),
            UNIQUE (tenant_id, document_id, ordinal),
            FOREIGN KEY (tenant_id, document_id)
                REFERENCES documents(tenant_id, document_id) ON DELETE CASCADE
        ) PARTITION BY HASH (tenant_id)
    """)
    for partition in range(16):
        op.execute(
            f"CREATE TABLE document_chunks_p{partition} PARTITION OF document_chunks "
            f"FOR VALUES WITH (MODULUS 16, REMAINDER {partition})"
        )

    op.execute("""
        CREATE TABLE document_acl (
            tenant_id TEXT NOT NULL,
            document_id UUID NOT NULL,
            principal_type TEXT NOT NULL CHECK (principal_type IN ('user', 'group', 'role')),
            principal_id TEXT NOT NULL,
            permission TEXT NOT NULL CHECK (permission IN ('read', 'write', 'admin')),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, document_id, principal_type, principal_id, permission),
            FOREIGN KEY (tenant_id, document_id)
                REFERENCES documents(tenant_id, document_id) ON DELETE CASCADE
        )
    """)
    op.execute("CREATE INDEX idx_document_acl_principal ON document_acl(tenant_id, principal_type, principal_id, permission)")
    op.execute("CREATE INDEX idx_document_chunks_embedding_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX idx_documents_metadata_gin ON documents USING gin (metadata)")
    op.execute("CREATE INDEX idx_document_chunks_metadata_gin ON document_chunks USING gin (metadata)")

    # RLS is enabled and forced on every tenant-owned table. The app must set
    # app.tenant_id with SET LOCAL inside each transaction after authentication.
    for table in ("tenants", "app_users", "sessions", "customer_tokens", "chat_history", "qa_templates", "documents", "document_chunks", "document_acl"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation_{table} ON {table} "
            "USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')) "
            "WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''))"
        )


def downgrade() -> None:
    for table in ("document_acl", "document_chunks", "documents", "qa_templates", "chat_history", "customer_tokens", "sessions", "app_users", "tenants"):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    op.execute("DROP EXTENSION IF EXISTS pgcrypto")
    op.execute("DROP EXTENSION IF EXISTS vector")
