"""Tenant groups and per-user membership for ACL-aware API identities.

Revision ID: pg0007
Revises: pg0006
"""
from alembic import op

revision = "pg0007"
down_revision = "pg0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE tenant_groups (
            tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            group_id BIGINT GENERATED ALWAYS AS IDENTITY,
            name TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, group_id),
            UNIQUE (tenant_id, name)
        )
    """)
    op.execute("""
        CREATE TABLE app_user_groups (
            tenant_id TEXT NOT NULL,
            user_id UUID NOT NULL,
            group_id BIGINT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, user_id, group_id),
            FOREIGN KEY (tenant_id, user_id)
                REFERENCES app_users(tenant_id, user_id) ON DELETE CASCADE,
            FOREIGN KEY (tenant_id, group_id)
                REFERENCES tenant_groups(tenant_id, group_id) ON DELETE CASCADE
        )
    """)
    for table in ("tenant_groups", "app_user_groups"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation_{table} ON {table} "
            "USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')) "
            "WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''))"
        )
    op.execute("GRANT SELECT ON tenant_groups, app_user_groups TO tenantrag_runtime")
    op.execute("GRANT USAGE, SELECT ON SEQUENCE tenant_groups_group_id_seq TO tenantrag_runtime")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS app_user_groups CASCADE")
    op.execute("DROP TABLE IF EXISTS tenant_groups CASCADE")
