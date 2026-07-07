"""
SessionService.

Replaces the logic that used to live directly inside main.py's
create_session() endpoint function: generating IDs/tokens, inserting into
two tables, and building the response — all inline, with no way to unit
test it without spinning up FastAPI and a real database connection.

Fix folded in here (see core/security.py for rationale): tokens are hashed
before being handed to the repositories, so nothing downstream ever touches
or stores a plaintext token.
"""
import uuid
from dataclasses import dataclass

from app.core.exceptions import NotFoundError
from app.core.security import generate_token, hash_token
from app.repositories.session_repository import SessionRepository
from app.repositories.tenant_repository import TenantRepository
from app.repositories.token_repository import TokenRepository


@dataclass
class NewSession:
    session_id: str
    tenant_id: str
    customer_id: str
    session_api_token: str
    customer_token: str


class SessionService:
    def __init__(
        self,
        session_repo: SessionRepository,
        token_repo: TokenRepository,
        tenant_repo: TenantRepository,
        session_token_ttl_hours: int,
        customer_token_ttl_days: int,
    ):
        self._session_repo = session_repo
        self._token_repo = token_repo
        self._tenant_repo = tenant_repo
        self._session_token_ttl_hours = session_token_ttl_hours
        self._customer_token_ttl_days = customer_token_ttl_days

    def create_session(self, tenant_id: str, customer_id: str) -> NewSession:
        # sessions.tenant_id is now a foreign key into tenants (migration
        # 0002) -- without this check, an unregistered tenant_id would
        # fail as an opaque MySQL FK-violation 500 instead of a clear,
        # actionable error.
        if not self._tenant_repo.exists(tenant_id):
            raise NotFoundError(
                f"Unknown tenant_id: {tenant_id!r}. Tenants must be registered before creating sessions.",
                error_code="UNKNOWN_TENANT",
            )

        session_id = str(uuid.uuid4())
        session_api_token = generate_token()
        customer_token = generate_token()
        customer_id_str = str(customer_id)

        # NOTE: the original code inserted into `sessions` and
        # `customer_tokens` as two separate statements without a shared
        # transaction. True transactional atomicity across both inserts
        # is a real gap -- if the second insert fails, the first is left
        # committed with no compensating action. Not fixed in this pass
        # (would require moving both inserts into one repository method
        # sharing a connection/transaction); flagged here rather than
        # silently left unmentioned.
        self._session_repo.create(
            session_id=session_id,
            tenant_id=tenant_id,
            customer_id=customer_id_str,
            api_token_hash=hash_token(session_api_token),
            ttl_hours=self._session_token_ttl_hours,
        )
        self._token_repo.create(
            token_hash=hash_token(customer_token),
            tenant_id=tenant_id,
            customer_id=customer_id_str,
            ttl_days=self._customer_token_ttl_days,
        )

        return NewSession(
            session_id=session_id,
            tenant_id=tenant_id,
            customer_id=customer_id_str,
            session_api_token=session_api_token,
            customer_token=customer_token,
        )
