from fastapi import APIRouter, Depends

from app.api.deps import get_session_service
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
    session_service: SessionService = Depends(get_session_service),
):
    # Blank-field validation now happens in the schema (see
    # app/schemas/session.py's field_validator) instead of an inline `if`
    # at the top of this function.
    new_session = session_service.create_session(request.tenant_id, request.customer_id)
    return SessionResponse(
        session_id=new_session.session_id,
        tenant_id=new_session.tenant_id,
        customer_id=new_session.customer_id,
        session_api_token=new_session.session_api_token,
        customer_token=new_session.customer_token,
    )
