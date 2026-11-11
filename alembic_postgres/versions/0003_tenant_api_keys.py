"""Allow tenant API keys in the pre-RLS authentication index.

Revision ID: pg0003
Revises: pg0002
"""
from alembic import op

revision = "pg0003"
down_revision = "pg0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE auth_token_lookup DROP CONSTRAINT auth_token_lookup_token_type_check")
    op.execute("""
        ALTER TABLE auth_token_lookup
        ADD CONSTRAINT auth_token_lookup_token_type_check
        CHECK (token_type IN ('session', 'customer', 'tenant_api_key'))
    """)
    op.execute("""
        CREATE INDEX idx_auth_token_lookup_tenant_key
        ON auth_token_lookup(token_hash)
        WHERE token_type = 'tenant_api_key' AND revoked_at IS NULL
    """)


def downgrade() -> None:
    op.execute("""
        DELETE FROM auth_token_lookup WHERE token_type = 'tenant_api_key'
    """)
    op.execute("DROP INDEX IF EXISTS idx_auth_token_lookup_tenant_key")
    op.execute("ALTER TABLE auth_token_lookup DROP CONSTRAINT auth_token_lookup_token_type_check")
    op.execute("""
        ALTER TABLE auth_token_lookup
        ADD CONSTRAINT auth_token_lookup_token_type_check
        CHECK (token_type IN ('session', 'customer'))
    """)
