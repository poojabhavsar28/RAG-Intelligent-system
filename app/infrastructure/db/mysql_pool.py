"""
MySQL connection pool.

Why this exists as its own module: the original main.py created a pool named
"mysql_pool" and AIAssistant.__init__ independently created a second pool
named "ai_mysql_pool" with a different size, both talking to the same
database. That's double the connections for no benefit, and two places that
can each fail independently at startup. There is now exactly one pool,
owned by the app lifespan and injected into repositories.
"""
import logging
from contextlib import contextmanager

from mysql.connector import pooling, Error

from app.core.config import Settings
from app.core.exceptions import DependencyUnavailableError

logger = logging.getLogger(__name__)


class MySQLPool:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._pool: pooling.MySQLConnectionPool | None = None

    def init(self) -> None:
        try:
            self._pool = pooling.MySQLConnectionPool(
                pool_name="app_mysql_pool",
                pool_size=self._settings.mysql_pool_size,
                pool_reset_session=True,
                host=self._settings.mysql_host,
                port=self._settings.mysql_port,
                user=self._settings.mysql_user,
                password=self._settings.mysql_password,
                database=self._settings.mysql_db,
                autocommit=True,
            )
            logger.info("MySQL connection pool initialized (size=%s)", self._settings.mysql_pool_size)
        except Error as e:
            logger.error(f"Failed to initialize MySQL connection pool: {e}")
            raise DependencyUnavailableError("Database is unavailable", error_code="DB_POOL_INIT_FAILED") from e

    def shutdown(self) -> None:
        if self._pool:
            try:
                self._pool._remove_connections()
            except Exception as e:  # pragma: no cover - best-effort cleanup
                logger.warning(f"Error during pool shutdown: {e}")

    @property
    def raw_pool(self) -> pooling.MySQLConnectionPool:
        """Exposes the underlying mysql.connector pool so it can be injected
        into code that expects a raw pool directly (e.g. the legacy
        AIAssistant class) instead of that code creating its own second
        pool against the same database."""
        if not self._pool:
            raise DependencyUnavailableError("Database connection pool not initialized", error_code="DB_POOL_NOT_INIT")
        return self._pool

    @contextmanager
    def connection(self):
        """Context manager so callers can't forget to close a connection
        (the original code had several endpoints that returned early on an
        exception path without closing the cursor/connection)."""
        if not self._pool:
            raise DependencyUnavailableError("Database connection pool not initialized", error_code="DB_POOL_NOT_INIT")
        conn = None
        try:
            conn = self._pool.get_connection()
            yield conn
        except Error as e:
            logger.error(f"Database error: {e}")
            raise DependencyUnavailableError("Database operation failed", error_code="DB_ERROR") from e
        finally:
            if conn is not None and conn.is_connected():
                conn.close()
