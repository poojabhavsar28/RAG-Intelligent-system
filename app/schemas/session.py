"""
Request/response schemas for session endpoints.

Pulled out of main.py verbatim (same fields, same types) so the API contract
doesn't change — only where the models live changes.
"""
from typing import Union

from pydantic import BaseModel, field_validator


class SessionCreateRequest(BaseModel):
    tenant_id: str
    customer_id: Union[str, int]

    @field_validator("tenant_id")
    @classmethod
    def tenant_id_not_blank(cls, v: str) -> str:
        # The original code checked `if not request.tenant_id` inside the
        # endpoint body. Moving it into the schema means FastAPI rejects the
        # request with a 422 before the endpoint function ever runs.
        if not v or not v.strip():
            raise ValueError("tenant_id must not be blank")
        return v


class SessionResponse(BaseModel):
    session_id: str
    tenant_id: str
    customer_id: str
    session_api_token: str
    customer_token: str
