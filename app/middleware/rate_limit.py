from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)


def get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For behind Render's proxy."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


settings = get_settings()

limiter = Limiter(
    key_func=get_client_ip,
    default_limits=[settings.rate_limit_default],
    storage_uri="memory://",
)
