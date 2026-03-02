from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.api.chat import router as chat_router
from app.api.evaluation import router as eval_router
from app.api.governance import router as governance_router
from app.api.agents import router as agents_router
from app.config import get_settings
from app.core.cache import get_cache_stats
from app.middleware.rate_limit import limiter
from app.middleware.security import SecurityHeadersMiddleware, RequestIDMiddleware
from app.observability.cost_tracker import usage_tracker
from app.observability.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

settings = get_settings()

allowed_origins = [settings.frontend_url.rstrip("/")]
if settings.local_frontend_url:
    allowed_origins.append(settings.local_frontend_url.rstrip("/"))

app = FastAPI(
    title="Rocket Mortgage FAQ Bot",
    description="RAG-powered FAQ assistant for Rocket Mortgage",
    version="1.0.0",
)

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware (added in reverse order: last added runs first)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID", "X-API-Key"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)

app.include_router(chat_router)
app.include_router(eval_router)
app.include_router(governance_router)
app.include_router(agents_router)


@app.get("/health")
async def health():
    return {"status": "ok", "cache": get_cache_stats()}


@app.get("/stats")
async def stats():
    return {
        "usage": usage_tracker.get_usage_summary(),
        "cache": get_cache_stats(),
    }


@app.on_event("startup")
async def startup():
    logger.info("application_starting")
