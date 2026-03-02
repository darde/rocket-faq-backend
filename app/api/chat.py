import re

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from app.config import get_settings
from app.core.llm import BudgetExceededError
from app.core.rag import generate_answer
from app.middleware.auth import verify_api_key
from app.middleware.rate_limit import limiter
from app.observability.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        v = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", v)
        return v.strip()


class SourceInfo(BaseModel):
    id: str
    score: float
    section: str
    subsection: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    query: str


@router.post("/", response_model=ChatResponse)
@limiter.limit(settings.rate_limit_chat)
async def chat(
    request: Request,
    body: ChatRequest,
    _api_key: str | None = Depends(verify_api_key),
):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info("chat_request", question=body.question[:100])

    try:
        result = generate_answer(body.question, top_k=body.top_k)
        return ChatResponse(
            answer=result.answer,
            sources=[SourceInfo(**s) for s in result.sources],
            query=result.query,
        )
    except BudgetExceededError:
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable due to usage limits. Please try again later.",
        )
    except Exception as e:
        logger.error("chat_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
