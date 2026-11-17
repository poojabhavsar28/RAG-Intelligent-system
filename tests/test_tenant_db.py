import pytest

from app.infrastructure.postgres.tenant_db import validate_tenant_id


@pytest.mark.parametrize("tenant_id", ["", "   ", None, 123, [], {}])
def test_validate_tenant_id_rejects_missing_or_non_string_values(tenant_id):
    with pytest.raises(ValueError, match="tenant_id must be a non-empty string"):
        validate_tenant_id(tenant_id)


@pytest.mark.parametrize(
    "tenant_id",
    ["tenant_123", "clinic-ahmedabad", "000123", "Tenant/Legacy-A"],
)
def test_validate_tenant_id_preserves_existing_identifier_exactly(tenant_id):
    assert validate_tenant_id(tenant_id) == tenant_id
