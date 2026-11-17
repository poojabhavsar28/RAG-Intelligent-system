"""Least-privilege PostgreSQL runtime role for TenantRAG.

Revision ID: pg0006
Revises: pg0005

The migration role remains privileged. Application connections must use a
login role that is a member of tenantrag_runtime (NOLOGIN, NOBYPASSRLS).
"""
from alembic import op

revision = "pg0006"
down_revision = "pg0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'tenantrag_runtime') THEN
            CREATE ROLE tenantrag_runtime NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
        END IF;
    END
    $$;
    """)
    op.execute("ALTER ROLE tenantrag_runtime NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS")
    op.execute("GRANT USAGE ON SCHEMA public TO tenantrag_runtime")
    op.execute("""
        GRANT SELECT, INSERT, UPDATE, DELETE ON
            tenants, app_users, sessions, customer_tokens, chat_history, qa_templates,
            documents, document_chunks, document_acl,
            acl_events, audit_log, feedback, usage_ledger
        TO tenantrag_runtime
    """)
    op.execute("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO tenantrag_runtime")
    op.execute("""
        GRANT EXECUTE ON FUNCTION public.lookup_auth_token(TEXT, TEXT, UUID)
        TO tenantrag_runtime
    """)
    op.execute("""
        GRANT EXECUTE ON FUNCTION public.lookup_session_tenant(UUID)
        TO tenantrag_runtime
    """)
    op.execute("""
        COMMENT ON ROLE tenantrag_runtime IS
        'TenantRAG application privilege group. Must not have SUPERUSER or BYPASSRLS.'
    """)


def downgrade() -> None:
    op.execute("REVOKE EXECUTE ON FUNCTION public.lookup_session_tenant(UUID) FROM tenantrag_runtime")
    op.execute("REVOKE EXECUTE ON FUNCTION public.lookup_auth_token(TEXT, TEXT, UUID) FROM tenantrag_runtime")
    op.execute("REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM tenantrag_runtime")
    op.execute("""
        REVOKE ALL ON
            tenants, app_users, sessions, customer_tokens, chat_history, qa_templates,
            documents, document_chunks, document_acl, auth_token_lookup,
            acl_events, audit_log, feedback, usage_ledger
        FROM tenantrag_runtime
    """)
    op.execute("REVOKE USAGE ON SCHEMA public FROM tenantrag_runtime")
    # The role may still have login members; deliberately do not DROP ROLE.
