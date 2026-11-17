"""add check constraints and expiry cleanup indexes

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-06

Two independent, low-risk additions:

1. CHECK constraints so invalid data is rejected by the database itself,
   not just by application-layer validation that every caller has to
   remember to run. Requires MySQL 8.0.16+ (CHECK constraints are parsed
   but silently not enforced on older versions -- confirmed target is
   8.0.16+ per this project's docker/deployment target).

2. Indexes on `expires_at` for sessions and customer_tokens. Neither
   table has this indexed yet, but an expired-session/token cleanup job
   is an obvious near-term need (there's currently no code path that ever
   deletes expired rows -- they just accumulate) and that job's query
   shape will be `WHERE expires_at < NOW()`, which needs this index to
   avoid a full table scan as these tables grow.
"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT chk_sessions_language "
        "CHECK (language IN ('en', 'hi'))"
    )
    op.execute(
        "ALTER TABLE qa_templates ADD CONSTRAINT chk_qa_templates_lang "
        "CHECK (lang IN ('en', 'hi'))"
    )
    op.execute(
        "ALTER TABLE chroma_metadata ADD CONSTRAINT chk_chroma_metadata_lang "
        "CHECK (lang IN ('en', 'hi'))"
    )

    op.execute("CREATE INDEX idx_sessions_expires_at ON sessions (expires_at)")
    op.execute("CREATE INDEX idx_customer_tokens_expires_at ON customer_tokens (expires_at)")


def downgrade() -> None:
    op.execute("DROP INDEX idx_customer_tokens_expires_at ON customer_tokens")
    op.execute("DROP INDEX idx_sessions_expires_at ON sessions")
    op.execute("ALTER TABLE chroma_metadata DROP CONSTRAINT chk_chroma_metadata_lang")
    op.execute("ALTER TABLE qa_templates DROP CONSTRAINT chk_qa_templates_lang")
    op.execute("ALTER TABLE sessions DROP CONSTRAINT chk_sessions_language")
