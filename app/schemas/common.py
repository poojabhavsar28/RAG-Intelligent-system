from typing import Any, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    database: str
    ai_assistant: str


class ErrorResponse(BaseModel):
    status_code: int
    detail: Any
    error: Optional[str] = None
