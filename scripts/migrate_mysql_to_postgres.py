"""One-way, restartable migration of legacy MySQL metadata into PostgreSQL.

Required environment:
  MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, [MYSQL_PORT]
  POSTGRES_ADMIN_DATABASE_URL

Run only after applying the PostgreSQL Alembic migrations. The target role
must have BYPASSRLS (normally a dedicated migration administrator). This
script preserves tenant IDs as exact strings and fails on non-UUID session
IDs rather than silently rewriting them.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Iterable
from uuid import UUID

import mysql.connector
from sqlalchemy import create_engine, text


BATCH_SIZE = 1000


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} must be configured")
    return value


def _tenant_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid legacy tenant_id: {value!r}")
    # Do not strip, lowercase, UUID-convert, or otherwise normalize IDs.
    return value


def _session_id(value: Any) -> UUID:
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(
            f"Legacy session_id {value!r} is not a UUID; migration stopped to avoid changing identity"
        ) from exc


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _batches(cursor, batch_size: int = BATCH_SIZE) -> Iterable[list[dict[str, Any]]]:
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        yield rows


def _assert_target_can_bypass_rls(connection) -> None:
    row = connection.execute(text("""
        SELECT r.rolsuper OR r.rolbypassrls AS can_bypass
        FROM pg_roles AS r
        WHERE r.rolname = current_user
    """)).scalar_one()
    if not row:
        raise RuntimeError(
            "POSTGRES_ADMIN_DATABASE_URL must use a migration role with BYPASSRLS; "
            "do not run a bulk import through the restricted application role"
        )


def migrate() -> dict[str, int]:
    mysql_conn = mysql.connector.connect(
        host=_required_env("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=_required_env("MYSQL_USER"),
        password=_required_env("MYSQL_PASSWORD"),
        database=_required_env("MYSQL_DB"),
    )
    pg_url = _required_env("POSTGRES_ADMIN_DATABASE_URL").replace(
        "postgresql://", "postgresql+psycopg://", 1
    )
    pg_engine = create_engine(pg_url, pool_pre_ping=True)
    counts = {"tenants": 0, "sessions": 0, "customer_tokens": 0, "chat_history": 0, "qa_templates": 0}

    try:
        with pg_engine.begin() as pg:
            _assert_target_can_bypass_rls(pg)

        # Tenants must exist before tenant-owned records are imported.
        source = mysql_conn.cursor(dictionary=True)
        try:
            source.execute("SELECT tenant_id, name, created_at FROM tenants ORDER BY tenant_id")
            for batch in _batches(source):
                payload = [{
                    "tenant_id": _tenant_id(row["tenant_id"]),
                    "name": row.get("name") or row["tenant_id"],
                    "created_at": _utc(row.get("created_at")),
                } for row in batch]
                with pg_engine.begin() as pg:
                    pg.execute(text("""
                        INSERT INTO tenants (tenant_id, name, created_at)
                        VALUES (:tenant_id, :name, :created_at)
                        ON CONFLICT (tenant_id) DO UPDATE
                        SET name = EXCLUDED.name
                    """), payload)
                counts["tenants"] += len(payload)
        finally:
            source.close()

        # Sessions precede chat history because chat_history has a composite FK.
        source = mysql_conn.cursor(dictionary=True)
        try:
            source.execute("""
                SELECT session_id, tenant_id, customer_id, api_token_hash,
                       language, created_at, expires_at
                FROM sessions ORDER BY session_id
            """)
            for batch in _batches(source):
                payload = [{
                    "session_id": _session_id(row["session_id"]),
                    "tenant_id": _tenant_id(row["tenant_id"]),
                    "customer_id": str(row["customer_id"]),
                    "token_hash": row["api_token_hash"],
                    "language": row.get("language") or "en",
                    "created_at": _utc(row.get("created_at")),
                    "expires_at": row.get("expires_at"),
                } for row in batch]
                with pg_engine.begin() as pg:
                    pg.execute(text("""
                        INSERT INTO sessions
                            (session_id, tenant_id, customer_id, token_hash, language, created_at, expires_at)
                        VALUES
                            (:session_id, :tenant_id, :customer_id, :token_hash, :language, :created_at, :expires_at)
                        ON CONFLICT (session_id) DO UPDATE SET
                            tenant_id = EXCLUDED.tenant_id,
                            customer_id = EXCLUDED.customer_id,
                            token_hash = EXCLUDED.token_hash,
                            language = EXCLUDED.language,
                            expires_at = EXCLUDED.expires_at
                    """), payload)
                    pg.execute(text("""
                        INSERT INTO auth_token_lookup
                            (token_hash, token_type, tenant_id, subject_id, session_id, expires_at)
                        VALUES
                            (:token_hash, 'session', :tenant_id, :customer_id, :session_id, :expires_at)
                        ON CONFLICT (token_hash) DO UPDATE SET
                            tenant_id = EXCLUDED.tenant_id,
                            subject_id = EXCLUDED.subject_id,
                            session_id = EXCLUDED.session_id,
                            expires_at = EXCLUDED.expires_at,
                            revoked_at = NULL
                    """), payload)
                counts["sessions"] += len(payload)
        finally:
            source.close()

        source = mysql_conn.cursor(dictionary=True)
        try:
            source.execute("""
                SELECT token_hash, tenant_id, customer_id, created_at, expires_at
                FROM customer_tokens ORDER BY token_hash
            """)
            for batch in _batches(source):
                payload = [{
                    "token_hash": row["token_hash"],
                    "tenant_id": _tenant_id(row["tenant_id"]),
                    "customer_id": str(row["customer_id"]),
                    "created_at": _utc(row.get("created_at")),
                    "expires_at": row.get("expires_at"),
                } for row in batch]
                with pg_engine.begin() as pg:
                    pg.execute(text("""
                        INSERT INTO customer_tokens
                            (token_hash, tenant_id, customer_id, created_at, expires_at)
                        VALUES
                            (:token_hash, :tenant_id, :customer_id, :created_at, :expires_at)
                        ON CONFLICT (token_hash) DO UPDATE SET
                            tenant_id = EXCLUDED.tenant_id,
                            customer_id = EXCLUDED.customer_id,
                            expires_at = EXCLUDED.expires_at
                    """), payload)
                    pg.execute(text("""
                        INSERT INTO auth_token_lookup
                            (token_hash, token_type, tenant_id, subject_id, expires_at)
                        VALUES
                            (:token_hash, 'customer', :tenant_id, :customer_id, :expires_at)
                        ON CONFLICT (token_hash) DO UPDATE SET
                            tenant_id = EXCLUDED.tenant_id,
                            subject_id = EXCLUDED.subject_id,
                            expires_at = EXCLUDED.expires_at,
                            revoked_at = NULL
                    """), payload)
                counts["customer_tokens"] += len(payload)
        finally:
            source.close()

        source = mysql_conn.cursor(dictionary=True)
        try:
            source.execute("""
                SELECT id, session_id, tenant_id, role, message, sender, timestamp
                FROM chat_history ORDER BY id
            """)
            for batch in _batches(source):
                payload = []
                for row in batch:
                    role = row["role"]
                    if role not in {"user", "assistant", "system"}:
                        raise ValueError(f"Unsupported chat_history role {role!r} at legacy id {row['id']}")
                    payload.append({
                        "id": int(row["id"]),
                        "session_id": _session_id(row["session_id"]),
                        "tenant_id": _tenant_id(row["tenant_id"]),
                        "role": role,
                        "message": row["message"],
                        "sender": row["sender"] or role,
                        "created_at": _utc(row.get("timestamp")),
                    })
                with pg_engine.begin() as pg:
                    pg.execute(text("""
                        INSERT INTO chat_history
                            (id, tenant_id, session_id, role, message, sender, created_at)
                        OVERRIDING SYSTEM VALUE
                        VALUES
                            (:id, :tenant_id, :session_id, :role, :message, :sender, :created_at)
                        ON CONFLICT (id) DO NOTHING
                    """), payload)
                counts["chat_history"] += len(payload)
        finally:
            source.close()

        source = mysql_conn.cursor(dictionary=True)
        try:
            source.execute("""
                SELECT id, tenant_id, lang, question, answer, created_at, updated_at
                FROM qa_templates ORDER BY id
            """)
            for batch in _batches(source):
                payload = [{
                    "id": int(row["id"]),
                    "tenant_id": _tenant_id(row["tenant_id"]),
                    "language": row.get("lang") or "en",
                    "question": row["question"],
                    "answer": row["answer"],
                    "created_at": _utc(row.get("created_at")),
                    "updated_at": _utc(row.get("updated_at")),
                } for row in batch]
                with pg_engine.begin() as pg:
                    pg.execute(text("""
                        INSERT INTO qa_templates
                            (id, tenant_id, language, question, answer, created_at, updated_at)
                        OVERRIDING SYSTEM VALUE
                        VALUES
                            (:id, :tenant_id, :language, :question, :answer, :created_at, :updated_at)
                        ON CONFLICT (id) DO UPDATE SET
                            tenant_id = EXCLUDED.tenant_id,
                            language = EXCLUDED.language,
                            question = EXCLUDED.question,
                            answer = EXCLUDED.answer,
                            updated_at = EXCLUDED.updated_at
                    """), payload)
                counts["qa_templates"] += len(payload)
        finally:
            source.close()

        # Explicit IDs were imported into identity columns; advance sequences.
        with pg_engine.begin() as pg:
            for table in ("chat_history", "qa_templates"):
                pg.execute(text(f"""
                    SELECT setval(
                        pg_get_serial_sequence('{table}', 'id'),
                        COALESCE((SELECT MAX(id) FROM {table}), 1),
                        EXISTS(SELECT 1 FROM {table})
                    )
                """))

        return counts
    finally:
        mysql_conn.close()
        pg_engine.dispose()


if __name__ == "__main__":
    result = migrate()
    print("MySQL to PostgreSQL migration counts:")
    for table, count in result.items():
        print(f"  {table}: {count}")
