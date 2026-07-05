from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.infrastructure.db.mysql_pool import MySQLPool


@dataclass
class TokenRecord:
    tenant_id: str
    customer_id: str


class TokenRepository:
    def __init__(self, pool: MySQLPool):
        self._pool = pool

    def create(self, token_hash: str, tenant_id: str, customer_id: str, ttl_days: int) -> None:
        expires_at = datetime.now(timezone.utc) + timedelta(days=ttl_days)
        with self._pool.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """INSERT INTO customer_tokens (token_hash, tenant_id, customer_id, expires_at)
                       VALUES (%s, %s, %s, %s)""",
                    (token_hash, tenant_id, customer_id, expires_at),
                )
                conn.commit()
            finally:
                cursor.close()

    def get_valid(self, token_hash: str) -> Optional[TokenRecord]:
        with self._pool.connection() as conn:
            cursor = conn.cursor(dictionary=True)
            try:
                cursor.execute(
                    """SELECT tenant_id, customer_id FROM customer_tokens
                       WHERE token_hash = %s AND (expires_at IS NULL OR expires_at > UTC_TIMESTAMP())""",
                    (token_hash,),
                )
                row = cursor.fetchone()
                return TokenRecord(**row) if row else None
            finally:
                cursor.close()
