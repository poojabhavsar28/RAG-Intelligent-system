"""baseline: schema as it existed before Alembic (matches schema.sql)

Revision ID: 0001
Revises:
Create Date: 2026-07-06

This migration intentionally reproduces the pre-Alembic schema as-is,
including its known gaps (no tenants table, no FK on chat_history's
CASCADE behavior specified). The fix for those gaps is 0002, kept
separate so this migration is an honest record of where the schema
actually started, not a retroactively "corrected" history.
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id VARCHAR(255) PRIMARY KEY,
            tenant_id VARCHAR(255) NOT NULL,
            customer_id VARCHAR(255) NOT NULL,
            api_token_hash VARCHAR(255) NOT NULL,
            language VARCHAR(10) DEFAULT 'en',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NULL,
            INDEX idx_tenant_customer (tenant_id, customer_id)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL,
            tenant_id VARCHAR(255) NOT NULL,
            role VARCHAR(50) NOT NULL,
            message TEXT NOT NULL,
            sender VARCHAR(50) NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id),
            INDEX idx_session_timestamp (session_id, timestamp)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS customer_tokens (
            token_hash VARCHAR(255) PRIMARY KEY,
            tenant_id VARCHAR(255) NOT NULL,
            customer_id VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NULL,
            INDEX idx_tenant_customer (tenant_id, customer_id)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS chroma_metadata (
            tenant_id VARCHAR(255) NOT NULL,
            lang VARCHAR(10) NOT NULL,
            db_path VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (tenant_id, lang)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS qa_templates (
            id INT AUTO_INCREMENT PRIMARY KEY,
            tenant_id VARCHAR(255) NOT NULL,
            lang VARCHAR(10) NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY unique_qa_per_tenant_lang (tenant_id, lang, question(255))
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS qa_templates")
    op.execute("DROP TABLE IF EXISTS chroma_metadata")
    op.execute("DROP TABLE IF EXISTS customer_tokens")
    op.execute("DROP TABLE IF EXISTS chat_history")
    op.execute("DROP TABLE IF EXISTS sessions")
