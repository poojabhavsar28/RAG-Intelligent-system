"""
Repository for the `sessions` table.

Why this exists: in the original main.py, endpoint functions
(create_session, get_questions, ask_question, get_chat_history) each opened
a connection and wrote raw SQL inline. That means the query for "find a
session by id" was duplicated across four endpoints with slightly different
column sets each time, and there was no single place to change it (e.g. to
add caching, or to add a WHERE deleted_at IS NULL clause later).

The repository pattern gives each table one owner. Endpoints call the
repository; they never see SQL.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.infrastructure.db.mysql_pool import MySQLPool


@dataclass
class SessionRecord:
    session_id: str
    tenant_id: str
    customer_id: str
    language: str = "en"


class SessionRepository:
    def __init__(self, pool: MySQLPool):
        self._pool = pool

    def create(self, session_id: str, tenant_id: str, customer_id: str, api_token_hash: str, ttl_hours: int) -> None:
        expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
        with self._pool.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """INSERT INTO sessions (session_id, tenant_id, customer_id, api_token_hash, expires_at)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (session_id, tenant_id, customer_id, api_token_hash, expires_at),
                )
                conn.commit()
            finally:
                cursor.close()

    def get_by_id(self, session_id: str) -> Optional[SessionRecord]:
        with self._pool.connection() as conn:
            cursor = conn.cursor(dictionary=True)
            try:
                cursor.execute(
                    "SELECT session_id, tenant_id, customer_id, language FROM sessions WHERE session_id = %s",
                    (session_id,),
                )
                row = cursor.fetchone()
                return SessionRecord(**row) if row else None
            finally:
                cursor.close()

    def get_by_id_and_token(self, session_id: str, api_token_hash: str) -> Optional[SessionRecord]:
        with self._pool.connection() as conn:
            cursor = conn.cursor(dictionary=True)
            try:
                cursor.execute(
                    """SELECT session_id, tenant_id, customer_id, language FROM sessions
                       WHERE session_id = %s AND api_token_hash = %s
                       AND (expires_at IS NULL OR expires_at > UTC_TIMESTAMP())""",
                    (session_id, api_token_hash),
                )
                row = cursor.fetchone()
                return SessionRecord(**row) if row else None
            finally:
                cursor.close()
