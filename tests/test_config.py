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


def test_settings_allow_default_mysql_values(monkeypatch):
    monkeypatch.delenv("MYSQL_USER", raising=False)
    monkeypatch.delenv("MYSQL_PASSWORD", raising=False)
    monkeypatch.delenv("MYSQL_DB", raising=False)

    settings = Settings()

    assert settings.mysql_user == "root"
    assert settings.mysql_password == "password"
    assert settings.mysql_db == "rag_chatbot"
