import logging
from pathlib import Path

from app.infrastructure.db.mysql_pool import MySQLPool

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def create_tables_if_not_exists(pool: MySQLPool) -> None:
    """Applies schema.sql. Fine for a portfolio project; Phase 2 replaces
    this with Alembic migrations so schema changes are versioned and
    reversible instead of being idempotent-CREATE-only."""
    sql_text = _SCHEMA_PATH.read_text()
    statements = []
    for raw_statement in sql_text.split(";"):
        statement = raw_statement.strip()
        if not statement or statement.startswith("--"):
            continue
        statements.append(statement)

    with pool.connection() as conn:
        cursor = conn.cursor()
        try:
            for statement in statements:
                cursor.execute(statement)
            conn.commit()
            logger.info("Database schema verified/created (%d statements)", len(statements))
        finally:
            cursor.close()
