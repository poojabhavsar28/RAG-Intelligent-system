"""
Token generation/hashing helpers.

The original schema stored api_token and customer_tokens.token as plaintext
VARCHAR columns compared with `WHERE api_token = %s`. Anyone with read access
to the database (a backup, a misconfigured replica, a careless query in a
BI tool) could impersonate any session. Storing a SHA-256 hash and comparing
hashes is a low-cost fix that doesn't change the API shape: the raw token is
still generated once and handed to the client, we just never persist it.

This is not a substitute for a real auth system (see security-and-performance
phase for a JWT/OAuth recommendation) but it's a meaningful improvement with
no added infrastructure.
"""
import hashlib
import secrets


def generate_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
