"""
Idempotent seed script for local development.

Safe to re-run: uses INSERT ... ON DUPLICATE KEY UPDATE, so running this
against a database that already has this data is a no-op rather than an
error or a duplicate row.

Usage:
    python scripts/seed_dev_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.infrastructure.db.mysql_pool import MySQLPool

SEED_TENANTS = [
    ("bob-retail", "Bank of Baroda \u2014 Retail", "Banking"),
    ("itms-fleet", "ITMS \u2014 Fleet Ops", "Logistics"),
    ("acme-insurance", "Acme Insurance (demo)", "Insurance"),
]


def main() -> None:
    settings = get_settings()
    pool = MySQLPool(settings)
    pool.init()

    with pool.connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.executemany(
                """
                INSERT INTO tenants (tenant_id, name, domain)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE name = VALUES(name), domain = VALUES(domain)
                """,
                SEED_TENANTS,
            )
            conn.commit()
            print(f"Seeded {len(SEED_TENANTS)} tenants.")
        finally:
            cursor.close()


if __name__ == "__main__":
    main()
