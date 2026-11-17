from dataclasses import dataclass
from typing import Optional

from app.infrastructure.db.mysql_pool import MySQLPool


@dataclass
class TenantRecord:
    tenant_id: str
    name: Optional[str]
    domain: Optional[str]


class TenantRepository:
    def __init__(self, pool: MySQLPool):
        self._pool = pool

    def get_by_id(self, tenant_id: str) -> Optional[TenantRecord]:
        with self._pool.connection() as conn:
            cursor = conn.cursor(dictionary=True)
            try:
                cursor.execute(
                    "SELECT tenant_id, name, domain FROM tenants WHERE tenant_id = %s",
                    (tenant_id,),
                )
                row = cursor.fetchone()
                return TenantRecord(**row) if row else None
            finally:
                cursor.close()

    def exists(self, tenant_id: str) -> bool:
        return self.get_by_id(tenant_id) is not None
