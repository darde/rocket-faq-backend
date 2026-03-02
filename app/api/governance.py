from fastapi import APIRouter, Depends, Request

from app.config import get_settings
from app.middleware.auth import verify_api_key
from app.middleware.rate_limit import limiter
from app.observability.audit import get_governance_summary
from app.observability.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/governance", tags=["governance"])


@router.get("/summary")
@limiter.limit(settings.rate_limit_default)
async def governance_summary(
    request: Request,
    _api_key: str | None = Depends(verify_api_key),
):
    logger.info("governance_summary_requested")
    return get_governance_summary()
