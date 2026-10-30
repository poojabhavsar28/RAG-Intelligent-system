"""Add a minimal global token index for pre-RLS authentication.

Revision ID: pg0002
Revises: pg0001
"""
from alembic import op

revision = "pg0002"
down_revision = "pg0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Authentication must resolve a tenant before the application can set
    # app.tenant_id. Keep only high-entropy token hashes and the minimum
    # routing metadata here; tenant-owned records remain protected by RLS.
    op.execute("""
        CREATE TABLE auth_token_lookup (
            token_hash TEXT PRIMARY KEY,
            token_type TEXT NOT NULL CHECK (token_type IN ('session', 'customer')),
            tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            subject_id TEXT NOT NULL,
            session_id UUID,
            expires_at TIMESTAMPTZ,
            revoked_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            FOREIGN KEY (tenant_id, session_id)
                REFERENCES sessions(tenant_id, session_id) ON DELETE CASCADE
        )
    """)
    op.execute("""
        CREATE INDEX idx_auth_token_lookup_session
        ON auth_token_lookup(session_id)
        WHERE token_type = 'session'
    """)
    op.execute("""
        CREATE INDEX idx_auth_token_lookup_tenant_subject
        ON auth_token_lookup(tenant_id, subject_id, token_type)
    """)
    op.execute("""
        COMMENT ON TABLE auth_token_lookup IS
        'Minimal hash-only authentication bootstrap index. Resolve tenant identity here, then perform tenant-owned access inside an RLS-scoped transaction. Never store plaintext tokens.'
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS auth_token_lookup")
