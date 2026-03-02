"""Optional API key authentication for endpoints."""

from __future__ import annotations

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str | None:
    """Verify API key if one is configured. Skip check if no key is set."""
    settings = get_settings()

    if not settings.api_key:
        return None

    if not api_key or api_key != settings.api_key:
        logger.warning("unauthorized_request", provided_key=bool(api_key))
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )

    return api_key
