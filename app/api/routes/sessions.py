from fastapi import APIRouter, Depends, Header, Request

from app.api.deps import get_session_service
from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.core.security import hash_token
from app.schemas.common import ErrorResponse
from app.schemas.session import SessionCreateRequest, SessionResponse
from app.services.session_service import SessionService

router = APIRouter(tags=["sessions"])


@router.post(
    "/create-session",
    response_model=SessionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Create a new session",
)
async def create_session(
    request: SessionCreateRequest,
    http_request: Request,
    tenant_api_key: str | None = Header(default=None, alias="X-Tenant-API-Key"),
    session_service: SessionService = Depends(get_session_service),
):
    # Blank-field validation now happens in the schema (see
    # app/schemas/session.py's field_validator) instead of an inline `if`
    # at the top of this function.
    if getattr(http_request.app.state, "database_backend", "mysql") == "postgres":
        if not tenant_api_key:
            raise UnauthorizedError(
                "X-Tenant-API-Key is required for PostgreSQL session creation",
                error_code="TENANT_API_KEY_REQUIRED",
            )
        trusted_tenant_id = http_request.app.state.tenant_repo.resolve_api_key(
            hash_token(tenant_api_key)
        )
        if not trusted_tenant_id:
            raise UnauthorizedError("Invalid tenant API key", error_code="INVALID_TENANT_API_KEY")
        if request.tenant_id != trusted_tenant_id:
            raise ForbiddenError(
                "Request tenant_id does not match the authenticated tenant",
                error_code="TENANT_ID_MISMATCH",
            )
    else:
        trusted_tenant_id = request.tenant_id

    new_session = session_service.create_session(trusted_tenant_id, request.customer_id)
    return SessionResponse(
        session_id=new_session.session_id,
        tenant_id=new_session.tenant_id,
        customer_id=new_session.customer_id,
        session_api_token=new_session.session_api_token,
        customer_token=new_session.customer_token,
    )
