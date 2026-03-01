from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.rag import generate_answer
from app.observability.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    top_k: int | None = None


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
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info("chat_request", question=request.question[:100])

    try:
        result = generate_answer(request.question, top_k=request.top_k)
        return ChatResponse(
            answer=result.answer,
            sources=[SourceInfo(**s) for s in result.sources],
            query=result.query,
        )
    except Exception as e:
        logger.error("chat_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
