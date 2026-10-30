"""
Application factory. This is the direct replacement for the old main.py.

Everything main.py did with module-level globals (db_pool, ai_assistant,
executor, reassigned inside an inline lifespan function) now happens here,
explicitly, with each piece stored on app.state instead of as a bare global.

IMPORTANT — migration note: this file constructs AIAssistant from the
*existing*, not-yet-refactored ai_assistant.py. Its ML logic (embeddings,
LLM loading, retrieval chain) is untouched — only its MySQL pool creation
was changed, to accept this file's pool via injection instead of creating
a duplicate one internally (see ai_assistant.py's __init__ db_pool param
and MySQLPool.raw_pool). Full migration into app/infrastructure/llm/ is a
separate, later phase — deferred deliberately (strangler-fig pattern:
decouple via the RAGEngine protocol first, migrate the implementation
only when there's a concrete reason to).
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, health, sessions
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging_config import configure_logging
from app.infrastructure.db.mysql_pool import MySQLPool
from app.infrastructure.postgres.tenant_db import create_postgres_engine
from app.infrastructure.postgres.repositories import (
    PostgresChatRepository,
    PostgresQATemplateRepository,
    PostgresSessionRepository,
    PostgresTenantRepository,
    PostgresTokenRepository,
)
from app.infrastructure.vectorstore.pgvector_store import PgVectorRetriever
from app.repositories.chat_repository import ChatRepository
from app.repositories.session_repository import SessionRepository
from app.repositories.tenant_repository import TenantRepository
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

    mysql_pool = None
    postgres_engine = None

    # Schema management is an explicit deployment step. The PostgreSQL
    # backend is opt-in until the metadata and Chroma importers have been
    # run and verified for the target environment.
    if settings.database_backend == "postgres":
        postgres_engine = create_postgres_engine(
            settings.postgres_database_url,
            pool_size=settings.postgres_pool_size,
            max_overflow=settings.postgres_max_overflow,
            pool_timeout=settings.postgres_pool_timeout,
        )
        session_repo = PostgresSessionRepository(postgres_engine)
        token_repo = PostgresTokenRepository(postgres_engine)
        chat_repo = PostgresChatRepository(postgres_engine)
        tenant_repo = PostgresTenantRepository(postgres_engine)
        qa_repository = PostgresQATemplateRepository(postgres_engine)

        from langchain_ollama import OllamaEmbeddings
        embedder = OllamaEmbeddings(model=settings.embedding_model)
        pg_retriever = PgVectorRetriever(postgres_engine, embedder)

        from ai_assistant import AIAssistant
        ai_assistant = AIAssistant(
            chroma_db_base_dir=settings.chroma_db_base_dir,
            executor=executor,
            qa_repository=qa_repository,
            pg_retriever=pg_retriever,
            disable_mysql=True,
        )
        app.state.database_backend = "postgres"
        app.state.postgres_engine = postgres_engine
        app.state.tenant_repo = tenant_repo
    else:
        mysql_pool = MySQLPool(settings)
        mysql_pool.init()
        session_repo = SessionRepository(mysql_pool)
        token_repo = TokenRepository(mysql_pool)
        chat_repo = ChatRepository(mysql_pool)
        tenant_repo = TenantRepository(mysql_pool)

        from ai_assistant import AIAssistant
        ai_assistant = AIAssistant(
            chroma_db_base_dir=settings.chroma_db_base_dir,
            executor=executor,
            db_pool=mysql_pool.raw_pool,
        )
        app.state.database_backend = "mysql"
        app.state.mysql_pool = mysql_pool
        app.state.tenant_repo = tenant_repo

    asyncio.create_task(_initialize_ai_assistant(ai_assistant))

    session_service = SessionService(
        session_repo, token_repo, tenant_repo, settings.session_token_ttl_hours, settings.customer_token_ttl_days
    )
    chat_service = ChatService(session_repo, token_repo, chat_repo, ai_assistant)
    qa_service = QAService(session_repo, ai_assistant)

    app.state.settings = settings
    app.state.executor = executor
    if mysql_pool is not None:
        app.state.mysql_pool = mysql_pool
    if postgres_engine is not None:
        app.state.postgres_engine = postgres_engine
    app.state.ai_assistant = ai_assistant
    app.state.session_service = session_service
    app.state.chat_service = chat_service
    app.state.qa_service = qa_service

    try:
        yield
    finally:
        logger.info("Shutting down application...")
        executor.shutdown(wait=False)
        if mysql_pool is not None:
            mysql_pool.shutdown()
        if postgres_engine is not None:
            postgres_engine.dispose()
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
