from __future__ import annotations

import hashlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tenantrag.api.production import (
    LocalCostLedger,
    RedisGateway,
    create_production_app,
    resolve_principal,
)


class FakeRedis:
    def __init__(self):
        self.calls = []

    def eval(self, script, numkeys, *args):
        self.calls.append((script, numkeys, args))
        if "DECR" in script:
            return 1
        return [1, b"ok", 0]

    def ping(self):
        return True

    def close(self):
        return None


class FakeConnection:
    def __init__(self, row):
        self.row = row
        self.last_sql = None
        self.last_params = None

    def execute(self, statement, params):
        self.last_sql = str(statement)
        self.last_params = params
        return SimpleNamespace(mappings=lambda: SimpleNamespace(first=lambda: self.row))


class FakeEngine:
    def __init__(self, row=None):
        self.connection = FakeConnection(row)

    @contextmanager
    def connect(self):
        yield self.connection


class FakeEmbedder:
    dim = 768

    def embed(self, texts):
        raise AssertionError("No model calls should happen in this test")

    def embed_query(self, text):
        raise AssertionError("No model calls should happen in this test")


class FakeLLM:
    name = "fake"

    def complete(self, system, prompt):
        raise AssertionError("No model calls should happen in this test")


def test_tenant_api_key_is_hashed_and_resolves_server_side_identity():
    raw_key = "tr_" + "x" * 48
    engine = FakeEngine({"tenant_id": "tenant-a", "subject_id": "tenant-a"})
    principal = resolve_principal(engine, raw_key)
    assert principal.tenant_id == "tenant-a"
    assert principal.user_id == 0
    assert principal.group_ids == ()
    assert engine.connection.last_params["token_hash"] == hashlib.sha256(raw_key.encode()).hexdigest()
    assert "lookup_auth_token" in engine.connection.last_sql
    assert raw_key not in engine.connection.last_sql


def test_invalid_tenant_key_is_rejected():
    with pytest.raises(HTTPException) as error:
        resolve_principal(FakeEngine(None), "tr_" + "x" * 48)
    assert error.value.status_code == 401


def test_short_api_key_is_rejected_without_database_lookup():
    with pytest.raises(HTTPException) as error:
        resolve_principal(FakeEngine(None), "short")
    assert error.value.status_code == 401


def test_redis_gateway_uses_atomic_shared_admission_and_release():
    client = FakeRedis()
    gateway = RedisGateway(client, rate_per_s=3, burst=9, max_inflight=2, daily_quota=100)
    admitted, reason, retry, tag = gateway.admit("tenant-a")
    assert admitted and reason == "ok" and retry == 0
    script, numkeys, args = client.calls[0]
    assert numkeys == 3
    assert "INCR" in script and "inflight" not in script  # inflight is an atomic Redis key
    assert args[0].startswith("trag:{")
    gateway.release(tag)
    assert len(client.calls) == 2
    assert "DECR" in client.calls[1][0]


def test_local_cost_ledger_does_not_claim_cloud_api_spend():
    ledger = LocalCostLedger()
    usd = ledger.record("tenant-a", 100, 200, 3)
    assert usd == 0.0
    usage = ledger.usage("tenant-a")
    assert usage.requests == 1
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 200
    assert usage.usd == 0.0


def test_production_app_has_no_demo_token_or_synthetic_corpus_route():
    app = create_production_app(FakeEngine(), FakeRedis(), FakeEmbedder(), FakeLLM())
    paths = {route.path for route in app.routes}
    assert {"/livez", "/healthz", "/v1/documents", "/v1/ask", "/v1/feedback", "/v1/usage"} <= paths
    assert "/demo" not in paths
    with TestClient(app) as client:
        assert client.get("/livez").json() == {"status": "alive"}
        response = client.post("/v1/ask", json={"query": "test"})
        assert response.status_code == 401
