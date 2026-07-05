-- Schema for the RAG chatbot service.
--
-- This consolidates table definitions that were previously duplicated
-- verbatim in main.py's create_tables_if_not_exists() and
-- ai_assistant.py's AIAssistant.init_mysql_db() (with drift risk: the two
-- copies could be edited independently and go out of sync).
--
-- Phase 2 (database design) replaces this with proper Alembic migrations;
-- this file is the starting point / source of truth for that migration.

CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    customer_id VARCHAR(255) NOT NULL,
    api_token_hash VARCHAR(255) NOT NULL,   -- hashed, not plaintext (see security phase)
    language VARCHAR(10) DEFAULT 'en',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NULL,
    INDEX idx_tenant_customer (tenant_id, customer_id)
);

CREATE TABLE IF NOT EXISTS chat_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    sender VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    INDEX idx_session_timestamp (session_id, timestamp)
);

CREATE TABLE IF NOT EXISTS customer_tokens (
    token_hash VARCHAR(255) PRIMARY KEY,    -- hashed, not plaintext
    tenant_id VARCHAR(255) NOT NULL,
    customer_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NULL,
    INDEX idx_tenant_customer (tenant_id, customer_id)
);

CREATE TABLE IF NOT EXISTS chroma_metadata (
    tenant_id VARCHAR(255) NOT NULL,
    lang VARCHAR(10) NOT NULL,
    db_path VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (tenant_id, lang)
);

CREATE TABLE IF NOT EXISTS qa_templates (
    id INT AUTO_INCREMENT PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    lang VARCHAR(10) NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_qa_per_tenant_lang (tenant_id, lang, question(255))
);
