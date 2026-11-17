import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache(monkeypatch):
    monkeypatch.delenv("MYSQL_USER", raising=False)
    monkeypatch.delenv("MYSQL_PASSWORD", raising=False)
    monkeypatch.delenv("MYSQL_DB", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_require_mysql_credentials_when_mysql_backend(monkeypatch):
    monkeypatch.delenv("MYSQL_USER", raising=False)
    monkeypatch.delenv("MYSQL_PASSWORD", raising=False)
    monkeypatch.delenv("MYSQL_DB", raising=False)

    # No credentials are baked into the code: the MySQL backend fails fast without them.
    with pytest.raises(ValueError, match="Missing MySQL configuration"):
        Settings(_env_file=None, database_backend="mysql")
