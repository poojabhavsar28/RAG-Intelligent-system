"""Application lifespan wiring tests for the PostgreSQL backend."""
from __future__ import annotations

import asyncio
import importlib
import sys
import types
from types import SimpleNamespace


def test_postgres_lifespan_wires_real_repository_types_and_disposes_engine(monkeypatch):
    # app.main creates its module-level app at import time. Supply a valid
    # configuration before importing it, then replace only infrastructure
    # boundaries so this test focuses on application wiring.
    monkeypatch.setenv("MYSQL_USER", "ci")
    monkeypatch.setenv("MYSQL_PASSWORD", "ci")
    monkeypatch.setenv("MYSQL_DB", "ci")

    import app.core.config as config
    config.get_settings.cache_clear()
    import app.main as main
    main = importlib.reload(main)

    from app.infrastructure.postgres.repositories import (
        PostgresChatRepository,
        PostgresQATemplateRepository,
        PostgresSessionRepository,
        PostgresTenantRepository,
        PostgresTokenRepository,
    )

    settings = SimpleNamespace(
        log_level="INFO",
        app_name="test",
        app_version="1",
        cors_allowed_origins=["http://localhost"],
        database_backend="postgres",
        postgres_database_url="postgresql://unused",
        postgres_pool_size=2,
        postgres_max_overflow=1,
        postgres_pool_timeout=5,
        max_workers=1,
        chroma_db_base_dir="./unused",
        embedding_model="test-embedding",
        session_token_ttl_hours=24,
        customer_token_ttl_days=30,
    )
    engine = SimpleNamespace(disposed=False)

    def dispose():
        engine.disposed = True

    engine.dispose = dispose
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "create_postgres_engine", lambda *args, **kwargs: engine)

    class FakeEmbeddings:
        def __init__(self, model):
            self.model = model

    class FakeAssistant:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def initialize_global_models(self):
            return None

    monkeypatch.setitem(
        sys.modules, "langchain_ollama",
        types.SimpleNamespace(OllamaEmbeddings=FakeEmbeddings),
    )
    monkeypatch.setitem(
        sys.modules, "ai_assistant",
        types.SimpleNamespace(AIAssistant=FakeAssistant),
    )
    monkeypatch.setattr(main, "_initialize_ai_assistant", lambda assistant: _noop())

    app = main.create_app()

    async def exercise_lifespan():
        async with main._lifespan(app):
            assert app.state.database_backend == "postgres"
            assert app.state.postgres_engine is engine
            assert isinstance(app.state.tenant_repo, PostgresTenantRepository)
            assert isinstance(app.state.session_service._session_repo, PostgresSessionRepository)
            assert isinstance(app.state.session_service._token_repo, PostgresTokenRepository)
            assert isinstance(app.state.chat_service._chat_repo, PostgresChatRepository)
            assert isinstance(app.state.qa_service._session_repo, PostgresSessionRepository)
            assert isinstance(app.state.ai_assistant, FakeAssistant)
            assert app.state.ai_assistant.kwargs["disable_mysql"] is True

    async def _run():
        await exercise_lifespan()

    asyncio.run(_run())
    assert engine.disposed is True


async def _noop():
    return None
