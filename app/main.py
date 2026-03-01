from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.evaluation import router as eval_router
from app.config import get_settings
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(eval_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.on_event("startup")
async def startup():
    logger.info("application_starting")
