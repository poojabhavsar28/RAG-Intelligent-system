from dataclasses import dataclass
from datetime import datetime
from typing import List

from app.infrastructure.db.mysql_pool import MySQLPool


@dataclass
class ChatMessageRecord:
    role: str
    message: str
    sender: str
    timestamp: datetime


class ChatRepository:
    def __init__(self, pool: MySQLPool):
        self._pool = pool

    def append_exchange(self, session_id: str, tenant_id: str, question: str, answer: str) -> None:
        """Writes the user question and assistant answer as two rows.
        Called as a background task from the /ask endpoint so the client
        doesn't wait on a write it doesn't need the result of."""
        with self._pool.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(
                    """INSERT INTO chat_history (session_id, tenant_id, role, message, sender)
                       VALUES (%s, %s, %s, %s, %s)""",
                    [
                        (session_id, tenant_id, "user", question, "user"),
                        (session_id, tenant_id, "assistant", answer, "assistant"),
                    ],
                )
                conn.commit()
            finally:
                cursor.close()

    def get_history(self, session_id: str) -> List[ChatMessageRecord]:
        with self._pool.connection() as conn:
            cursor = conn.cursor(dictionary=True)
            try:
                cursor.execute(
                    """SELECT role, message, sender, timestamp FROM chat_history
                       WHERE session_id = %s ORDER BY timestamp ASC""",
                    (session_id,),
                )
                return [ChatMessageRecord(**row) for row in cursor.fetchall()]
            finally:
                cursor.close()
