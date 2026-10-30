import pytest

from app.infrastructure.postgres.tenant_db import tenant_transaction


@pytest.mark.parametrize("tenant_id", ["", "not-a-uuid", None, "1234"])
def test_tenant_transaction_rejects_invalid_tenant_before_db_access(tenant_id):
    with pytest.raises(ValueError, match="tenant_id must be a valid UUID"):
        with tenant_transaction(engine=None, tenant_id=tenant_id):
            pytest.fail("invalid tenant must not open a transaction")
