"""Restrict direct access to the pre-RLS token index.

Revision ID: pg0004
Revises: pg0003
"""
from alembic import op

revision = "pg0004"
down_revision = "pg0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE OR REPLACE FUNCTION public.lookup_auth_token(
            p_token_hash TEXT,
            p_token_type TEXT,
            p_session_id UUID DEFAULT NULL
        )
        RETURNS TABLE (tenant_id TEXT, subject_id TEXT, session_id UUID)
        LANGUAGE SQL
        STABLE
        SECURITY DEFINER
        SET search_path = pg_catalog, public
        AS $function$
            SELECT a.tenant_id, a.subject_id, a.session_id
            FROM public.auth_token_lookup AS a
            WHERE a.token_hash = p_token_hash
              AND a.token_type = p_token_type
              AND (p_session_id IS NULL OR a.session_id = p_session_id)
              AND a.revoked_at IS NULL
              AND (a.expires_at IS NULL OR a.expires_at > now())
            LIMIT 1
        $function$
    """)
    op.execute("""
        CREATE OR REPLACE FUNCTION public.lookup_session_tenant(p_session_id UUID)
        RETURNS TEXT
        LANGUAGE SQL
        STABLE
        SECURITY DEFINER
        SET search_path = pg_catalog, public
        AS $function$
            SELECT a.tenant_id
            FROM public.auth_token_lookup AS a
            WHERE a.session_id = p_session_id
              AND a.token_type = 'session'
              AND a.revoked_at IS NULL
              AND (a.expires_at IS NULL OR a.expires_at > now())
            ORDER BY a.created_at DESC
            LIMIT 1
        $function$
    """)
    op.execute("""
        CREATE OR REPLACE FUNCTION public.register_auth_token(
            p_token_hash TEXT,
            p_token_type TEXT,
            p_tenant_id TEXT,
            p_subject_id TEXT,
            p_session_id UUID,
            p_expires_at TIMESTAMPTZ
        )
        RETURNS VOID
        LANGUAGE SQL
        SECURITY DEFINER
        SET search_path = pg_catalog, public
        AS $function$
            INSERT INTO public.auth_token_lookup
                (token_hash, token_type, tenant_id, subject_id, session_id, expires_at)
            VALUES
                (p_token_hash, p_token_type, p_tenant_id, p_subject_id, p_session_id, p_expires_at)
        $function$
    """)
    op.execute("REVOKE ALL ON FUNCTION public.lookup_auth_token(TEXT, TEXT, UUID) FROM PUBLIC")
    op.execute("REVOKE ALL ON FUNCTION public.lookup_session_tenant(UUID) FROM PUBLIC")
    op.execute("REVOKE ALL ON FUNCTION public.register_auth_token(TEXT, TEXT, TEXT, TEXT, UUID, TIMESTAMPTZ) FROM PUBLIC")
    op.execute("REVOKE ALL ON TABLE public.auth_token_lookup FROM PUBLIC")


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS public.register_auth_token(TEXT, TEXT, TEXT, TEXT, UUID, TIMESTAMPTZ)")
    op.execute("DROP FUNCTION IF EXISTS public.lookup_session_tenant(UUID)")
    op.execute("DROP FUNCTION IF EXISTS public.lookup_auth_token(TEXT, TEXT, UUID)")
