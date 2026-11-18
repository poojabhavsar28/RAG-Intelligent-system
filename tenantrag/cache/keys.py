"""Per-tenant key derivation. A KMS would hold ``master``; here it is an injected secret.
Keys are never built by string concatenation: fields are length-prefixed, so a tenant id
or query containing a delimiter cannot alias another key."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import struct
from typing import Iterable


def canonical_encode(fields: Iterable[tuple[str, object]]) -> bytes:
    out = bytearray()
    for name, value in fields:
        for part in (name.encode(), str(value).encode()):
            out += struct.pack(">I", len(part)) + part
    return bytes(out)


def _b32(b: bytes) -> str:
    return base64.b32encode(b).decode().rstrip("=").lower()


class KeyService:
    def __init__(self, master: bytes | None = None):
        self._master = master or os.urandom(32)

    def _derive(self, purpose: bytes, tenant_id: str) -> bytes:
        return hmac.new(self._master, canonical_encode([("p", purpose.decode()), ("t", tenant_id)]), hashlib.sha256).digest()

    def mac_key(self, tenant_id: str) -> bytes:
        return self._derive(b"mac", tenant_id)

    def data_key(self, tenant_id: str) -> bytes:
        return self._derive(b"enc", tenant_id)

    def tenant_tag(self, tenant_id: str) -> str:
        return self._derive(b"tag", tenant_id).hex()[:12]  # raw tenant ids never appear in keys

    def global_mac_key(self) -> bytes:  # only used to model the "forgot the tenant" bug in tests
        return hmac.new(self._master, b"global", hashlib.sha256).digest()

    @staticmethod
    def mac(key: bytes, fields: Iterable[tuple[str, object]]) -> str:
        return _b32(hmac.new(key, canonical_encode(fields), hashlib.sha256).digest())
