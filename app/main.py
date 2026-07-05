"""
Application factory. This is the direct replacement for the old main.py.

Everything main.py did with module-level globals (db_pool, ai_assistant,
executor, reassigned inside an inline lifespan function) now happens here,
explicitly, with each piece stored on app.state instead of as a bare global.

IMPORTANT — migration note: this file constructs AIAssistant from the
*existing*, not-yet-refactored ai_assistant.py. That class still does its
own MySQL pool creation internally (see ai_assistant.py's __init__), which
duplicates the pool this file also creates. That duplication is called out
explicitly rather than silently kept: it gets removed once ai_assistant.py
itself is migrated into app/infrastructure/llm/ in the next phase, at which
point it will accept an injected MySQLPool instead of making its own.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, health, sessions
from app.core.config import get_settings
from app.core.exceptions import DependencyUnavailableError, register_exception_handlers
from app.core.logging_config import configure_logging
from app.infrastructure.db.bootstrap import create_tables_if_not_exists
from app.infrastructure.db.mysql_pool import MySQLPool
from app.repositories.chat_repository import ChatRepository
from app.repositories.session_repository import SessionRepository
from app.repositories.token_repository import TokenRepository
from app.services.chat_service import ChatService
from app.services.qa_service import QAService
from app.services.session_service import SessionService

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        # Was ["*"] in the original code, which combined with
        # allow_credentials=True is a real vulnerability (any site can make
        # authenticated requests on a logged-in user's behalf). Now sourced
        # from Settings so each environment declares its own allow-list.
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(chat.router)

    return app


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting application lifespan...")

    executor = ThreadPoolExecutor(max_workers=settings.max_workers, thread_name_prefix="ai_worker")
    logger.info("Thread pool executor initialized with %s workers", settings.max_workers)

    mysql_pool = MySQLPool(settings)
    try:
        mysql_pool.init()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, create_tables_if_not_exists, mysql_pool)
    except DependencyUnavailableError:
        logger.warning("MySQL is unavailable during startup; continuing without database initialization")

    session_repo = SessionRepository(mysql_pool)
    token_repo = TokenRepository(mysql_pool)
    chat_repo = ChatRepository(mysql_pool)

    # --- AI assistant (not yet migrated — see module docstring) ---
    from ai_assistant import AIAssistant  # local import: legacy module, migrates in the LLM-layer phase

    ai_assistant = AIAssistant(chroma_db_base_dir=settings.chroma_db_base_dir, executor=executor)
    asyncio.create_task(_initialize_ai_assistant(ai_assistant))

    session_service = SessionService(
        session_repo, token_repo, settings.session_token_ttl_hours, settings.customer_token_ttl_days
    )
    chat_service = ChatService(session_repo, token_repo, chat_repo, ai_assistant)
    qa_service = QAService(session_repo, ai_assistant)

    app.state.settings = settings
    app.state.executor = executor
    app.state.mysql_pool = mysql_pool
    app.state.ai_assistant = ai_assistant
    app.state.session_service = session_service
    app.state.chat_service = chat_service
    app.state.qa_service = qa_service

    try:
        yield
    finally:
        logger.info("Shutting down application...")
        executor.shutdown(wait=False)
        mysql_pool.shutdown()
        logger.info("Application shutdown complete")


async def _initialize_ai_assistant(ai_assistant) -> None:
    try:
        await ai_assistant.initialize_global_models()
        logger.info("AI Assistant initialized successfully")
    except Exception:
        logger.exception("Failed to initialize AI Assistant")


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1, loop="asyncio", log_config=None)
