"""Least-privilege runtime role is created by the PostgreSQL migration."""
import os

import pytest
from sqlalchemy import create_engine, text

pytestmark = pytest.mark.skipif(
    not os.getenv("POSTGRES_DATABASE_URL"),
    reason="POSTGRES_DATABASE_URL required",
)


def test_tenantrag_runtime_role_is_non_bypass_and_token_lookup_is_function_only():
    url = os.environ["POSTGRES_DATABASE_URL"].replace(
        "postgresql://", "postgresql+psycopg://", 1
    )
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            role = conn.execute(text("""
                SELECT rolsuper, rolbypassrls, rolcreatedb, rolcreaterole
                FROM pg_roles WHERE rolname='tenantrag_runtime'
            """)).one()
            assert tuple(role) == (False, False, False, False)
            assert conn.execute(text("""
                SELECT has_function_privilege(
                    'tenantrag_runtime',
                    'public.lookup_auth_token(text,text,uuid)',
                    'EXECUTE'
                )
            """)).scalar_one()
            assert conn.execute(text("""
                SELECT has_table_privilege(
                    'tenantrag_runtime', 'public.app_user_groups', 'SELECT'
                )
            """)).scalar_one()
            assert not conn.execute(text("""
                SELECT has_table_privilege(
                    'tenantrag_runtime', 'public.auth_token_lookup', 'SELECT'
                )
            """)).scalar_one()
    finally:
        engine.dispose()
