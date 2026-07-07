"""add tenants table, backfill, and explicit foreign keys

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-06

Fixes the core normalization gap in the baseline schema: tenant_id was a
free-text string repeated across five tables with no canonical source of
truth, so there was nothing preventing a typo'd tenant_id from silently
creating a "new" tenant. This migration:

1. Creates a `tenants` table
2. Backfills it from every distinct tenant_id already present across
   sessions, customer_tokens, chroma_metadata, and qa_templates
3. Adds foreign keys from those tables (and chat_history) to tenants,
   each with an ON DELETE behavior chosen deliberately per relationship
   (see comments below) rather than left to the database default
4. Fixes chat_history's FK to sessions, which previously had no explicit
   ON DELETE behavior (defaults to RESTRICT in InnoDB) -- changed to
   CASCADE, since chat history has no independent meaning once its
   session is gone
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE tenants (
            tenant_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NULL,
            domain VARCHAR(100) NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # Backfill: every tenant_id that already exists anywhere becomes a row
    # here. name/domain are left NULL -- there was no source of truth for
    # them before this table existed. A follow-up admin task (out of scope
    # for this migration) can populate them.
    op.execute(
        """
        INSERT IGNORE INTO tenants (tenant_id)
        SELECT DISTINCT tenant_id FROM sessions
        UNION
        SELECT DISTINCT tenant_id FROM customer_tokens
        UNION
        SELECT DISTINCT tenant_id FROM chroma_metadata
        UNION
        SELECT DISTINCT tenant_id FROM qa_templates
        """
    )

    # sessions -> tenants: RESTRICT. A tenant with active sessions
    # shouldn't be deletable without first resolving those sessions --
    # silent cascade-deleting a tenant's entire session history is too
    # destructive for an implicit foreign key action.
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT fk_sessions_tenant "
        "FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE RESTRICT"
    )

    # customer_tokens -> tenants: RESTRICT, same reasoning as sessions.
    op.execute(
        "ALTER TABLE customer_tokens ADD CONSTRAINT fk_customer_tokens_tenant "
        "FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE RESTRICT"
    )

    # chroma_metadata -> tenants: RESTRICT. This row points at a physical
    # vector store path on disk; deleting it as a side effect of a tenant
    # deletion would silently orphan that directory instead of triggering
    # an explicit cleanup step.
    op.execute(
        "ALTER TABLE chroma_metadata ADD CONSTRAINT fk_chroma_metadata_tenant "
        "FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE RESTRICT"
    )

    # qa_templates -> tenants: CASCADE. Unlike the above, these are
    # tenant-owned configuration content with no meaning outside that
    # tenant and no external side effects (no files on disk, no other
    # table references them) -- safe to delete along with the tenant.
    op.execute(
        "ALTER TABLE qa_templates ADD CONSTRAINT fk_qa_templates_tenant "
        "FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE"
    )

    # chat_history -> tenants: RESTRICT (consistent with sessions/tokens above).
    op.execute(
        "ALTER TABLE chat_history ADD CONSTRAINT fk_chat_history_tenant "
        "FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE RESTRICT"
    )

    # chat_history -> sessions: fix the missing ON DELETE behavior.
    # The original FK (created in 0001, matching the pre-Alembic schema)
    # had no explicit ON DELETE, which defaults to RESTRICT in InnoDB --
    # meaning a session could never actually be deleted while it had any
    # chat history, with no clear error explaining why. CASCADE is correct
    # here: chat history has no independent meaning once its session is gone.
    conn = op.get_bind()
    constraint_name = conn.execute(
        sa.text(
            """
            SELECT CONSTRAINT_NAME FROM information_schema.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'chat_history'
              AND COLUMN_NAME = 'session_id'
              AND REFERENCED_TABLE_NAME = 'sessions'
            LIMIT 1
            """
        )
    ).scalar()
    op.execute(f"ALTER TABLE chat_history DROP FOREIGN KEY {constraint_name}")
    op.execute(
        "ALTER TABLE chat_history ADD CONSTRAINT fk_chat_history_session "
        "FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE"
    )


def downgrade() -> None:
    conn = op.get_bind()
    constraint_name = conn.execute(
        sa.text(
            """
            SELECT CONSTRAINT_NAME FROM information_schema.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'chat_history'
              AND COLUMN_NAME = 'session_id'
              AND REFERENCED_TABLE_NAME = 'sessions'
            LIMIT 1
            """
        )
    ).scalar()
    op.execute(f"ALTER TABLE chat_history DROP FOREIGN KEY {constraint_name}")
    op.execute(
        "ALTER TABLE chat_history ADD CONSTRAINT chat_history_ibfk_1 "
        "FOREIGN KEY (session_id) REFERENCES sessions(session_id)"
    )

    op.execute("ALTER TABLE chat_history DROP FOREIGN KEY fk_chat_history_tenant")
    op.execute("ALTER TABLE qa_templates DROP FOREIGN KEY fk_qa_templates_tenant")
    op.execute("ALTER TABLE chroma_metadata DROP FOREIGN KEY fk_chroma_metadata_tenant")
    op.execute("ALTER TABLE customer_tokens DROP FOREIGN KEY fk_customer_tokens_tenant")
    op.execute("ALTER TABLE sessions DROP FOREIGN KEY fk_sessions_tenant")
    op.execute("DROP TABLE IF EXISTS tenants")
