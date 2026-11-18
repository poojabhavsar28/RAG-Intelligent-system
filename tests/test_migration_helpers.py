from datetime import datetime, timezone
from uuid import UUID

import pytest

from scripts.migrate_mysql_to_postgres import _session_id, _tenant_id, _utc


def test_migration_preserves_tenant_id_exactly():
    value = "Legacy/Tenant-Ahmedabad "
    assert _tenant_id(value) == value


@pytest.mark.parametrize("value", [None, "", "   ", 123])
def test_migration_rejects_invalid_tenant_ids(value):
    with pytest.raises(ValueError, match="Invalid legacy tenant_id"):
        _tenant_id(value)


def test_migration_parses_existing_uuid_session_ids_without_replacement():
    value = "12345678-1234-5678-1234-567812345678"
    assert _session_id(value) == UUID(value)


def test_migration_stops_on_non_uuid_session_ids():
    with pytest.raises(ValueError, match="avoid changing identity"):
        _session_id("legacy-session-42")


def test_migration_interprets_naive_mysql_timestamps_as_utc():
    value = datetime(2026, 10, 31, 12, 0, 0)
    assert _utc(value) == value.replace(tzinfo=timezone.utc)
