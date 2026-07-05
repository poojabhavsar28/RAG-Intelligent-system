"""
Centralized application configuration.

Why this exists: the original codebase called os.getenv() independently in
main.py, ai_assistant.py, and scanning.py, each with its own defaults (or no
default at all). That means the same setting could silently resolve to three
different values depending on which module read it first, and there was no
single place to see what configuration the app actually needs.

Trade-off: pydantic-settings adds a dependency, but it gets us validation
(fail fast at startup if a required var is missing) and type coercion for free.
"""
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- App ---
    app_name: str = "RAG Chatbot API"
    app_version: str = "1.0.0"
    environment: str = Field(default="development")  # development | staging | production
    log_level: str = "INFO"

    # --- CORS ---
    # NOTE: previously hardcoded to ["*"]. Must be an explicit allow-list in
    # production; wildcard + allow_credentials=True is a real vulnerability
    # (see security-and-performance phase).
    cors_allowed_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # --- MySQL ---
    mysql_host: str = Field(default="localhost")
    mysql_port: int = Field(default=3306)
    mysql_user: str = Field(default="root")
    mysql_password: str = Field(default="password")
    mysql_db: str = Field(default="rag_chatbot")
    mysql_pool_size: int = Field(default=10)

    # --- Vector store ---
    chroma_db_base_dir: str = "./chroma_db_base"

    # --- LLM / embeddings ---
    embedding_model: str = "nomic-embed-text"
    llm_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.3

    # --- Concurrency ---
    max_workers: int = 10

    # --- Auth token lifetime ---
    session_token_ttl_hours: int = 24
    customer_token_ttl_days: int = 30


@lru_cache
def get_settings() -> Settings:
    """Cached so Settings() is only constructed (and .env parsed) once per process."""
    return Settings()
