import re

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from app.config import get_settings
from app.core.llm import BudgetExceededError
from app.core.rag import generate_answer
from app.middleware.auth import verify_api_key
from app.middleware.rate_limit import limiter
from app.observability.audit import update_audit_feedback
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


class GuardrailsInfo(BaseModel):
    input_pii_detected: bool = False
    injection_detected: bool = False
    off_topic: bool = False
    low_confidence: bool = False
    disclaimers_added: list[str] = Field(default_factory=list)
    blocked: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    query: str
    guardrails: GuardrailsInfo
    request_id: str


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    rating: str = Field(..., pattern=r"^(positive|negative)$")
    comment: str | None = Field(default=None, max_length=500)


class FeedbackResponse(BaseModel):
    status: str
    request_id: str


@router.post("/", response_model=ChatResponse)
@limiter.limit(settings.rate_limit_chat)
async def chat(
    request: Request,
    body: ChatRequest,
    _api_key: str | None = Depends(verify_api_key),
):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    request_id = getattr(request.state, "request_id", "")
    logger.info("chat_request", question=body.question[:100])

    try:
        result = generate_answer(
            body.question, top_k=body.top_k, request_id=request_id
        )
        return ChatResponse(
            answer=result.answer,
            sources=[SourceInfo(**s) for s in result.sources],
            query=result.query,
            guardrails=GuardrailsInfo(**{
                k: v for k, v in result.guardrails.items()
                if k in GuardrailsInfo.model_fields
            }),
            request_id=request_id,
        )
    except BudgetExceededError:
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable due to usage limits. Please try again later.",
        )
    except Exception as e:
        logger.error("chat_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", response_model=FeedbackResponse)
@limiter.limit(settings.rate_limit_chat)
async def submit_feedback(
    request: Request,
    body: FeedbackRequest,
    _api_key: str | None = Depends(verify_api_key),
):
    logger.info("feedback_received", request_id=body.request_id, rating=body.rating)

    success = update_audit_feedback(body.request_id, body.rating, body.comment)
    if not success:
        raise HTTPException(status_code=404, detail="Request ID not found in audit log")

    return FeedbackResponse(status="recorded", request_id=body.request_id)
