"""Track document ownership for safe ACL updates and re-ingestion.

Revision ID: pg0008
Revises: pg0007
"""
from alembic import op

revision = "pg0008"
down_revision = "pg0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE documents
        ADD COLUMN created_by TEXT NOT NULL DEFAULT 'tenant-service'
    """)
    op.execute("""
        COMMENT ON COLUMN documents.created_by IS
        'Authenticated tenant user subject or tenant-service for service-key-created documents.'
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS created_by")
