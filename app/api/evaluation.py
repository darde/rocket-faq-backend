from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.vectorstore import search
from app.core.rag import generate_answer
from app.evaluation.metrics import evaluate_retrieval, mean_reciprocal_rank
from app.evaluation.judge import evaluate_response
from app.middleware.auth import verify_api_key
from app.middleware.rate_limit import limiter
from app.observability.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/eval", tags=["evaluation"])

EVAL_DATASET = [
    {
        "query": "How do I set up autopay?",
        "relevant_questions": [
            "How can I set up autopay?",
            "What are the benefits of autopay?",
        ],
    },
    {
        "query": "What happens to my escrow when I pay off my mortgage?",
        "relevant_questions": [
            "What happens to the money in my escrow account when my loan has been paid in full?",
            "Once my loan balance has been fully repaid, what happens to any money left in my escrow account?",
        ],
    },
    {
        "query": "How can I remove PMI from my loan?",
        "relevant_questions": [
            "Can I remove mortgage insurance (PMI)?",
            "Why do I need private mortgage insurance (PMI)?",
            "Does it matter how long I've been making payments on my loan?",
        ],
    },
    {
        "query": "When will I get my 1098 tax form?",
        "relevant_questions": [
            "When can I expect to receive my 1098 tax statement?",
            "How will I get my 1098?",
        ],
    },
    {
        "query": "What are the payment options available?",
        "relevant_questions": [
            "What are the different ways to make a payment?",
            "What options are available for making payments with Rocket Mortgage?",
        ],
    },
    {
        "query": "How do I file an insurance claim for property damage?",
        "relevant_questions": [
            "There's been damage to my property. How do I file an insurance claim, and how does the process work?",
            "I've received an insurance claim check made payable to Rocket Mortgage. What should I do with it?",
        ],
    },
    {
        "query": "Can I make biweekly mortgage payments?",
        "relevant_questions": [
            "How can I set up biweekly autopay?",
            "What are the benefits of paying my mortgage biweekly instead of monthly?",
            "How do biweekly payments work?",
        ],
    },
    {
        "query": "What is the forbearance program?",
        "relevant_questions": [
            "What are my options for catching up on payments that were paused?",
            "What if I still can't make payments after my first 3 months of forbearance?",
        ],
    },
    {
        "query": "How do I draw money from my HELOC?",
        "relevant_questions": [
            "How can I draw from my line of credit?",
            "What is the cutoff time for draw or transfer requests?",
        ],
    },
    {
        "query": "Why did my mortgage payment increase?",
        "relevant_questions": [
            "I have a fixed-rate mortgage. Why is my payment changing?",
            "An escrow analysis was just completed for my account and my payment went up. Why?",
        ],
    },
]


class EvalRetrievalRequest(BaseModel):
    k: int = Field(default=5, ge=1, le=20)


class SingleEvalResult(BaseModel):
    query: str
    metrics: dict
    retrieved_questions: list[str]
    relevant_questions: list[str]


class RetrievalEvalResponse(BaseModel):
    results: list[SingleEvalResult]
    aggregate: dict


class JudgeRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class JudgeResponse(BaseModel):
    question: str
    answer: str
    evaluation: dict
    sources: list[dict]


class FullEvalResponse(BaseModel):
    retrieval_metrics: dict
    judge_evaluations: list[dict]
    overall_summary: dict


@router.post("/retrieval", response_model=RetrievalEvalResponse)
@limiter.limit(settings.rate_limit_eval)
async def evaluate_retrieval_endpoint(
    request: Request,
    body: EvalRetrievalRequest = EvalRetrievalRequest(),
    _api_key: str | None = Depends(verify_api_key),
):
    """Evaluate retrieval quality using precision, recall, and MRR."""
    logger.info("retrieval_eval_start", k=body.k)

    results = []
    all_query_results = []

    for item in EVAL_DATASET:
        query = item["query"]
        relevant_qs = set(item["relevant_questions"])

        docs = search(query, top_k=body.k)

        retrieved_questions = [d["metadata"].get("question", "") for d in docs]
        retrieved_for_metrics = retrieved_questions

        relevant_for_metrics = relevant_qs

        metrics = evaluate_retrieval(
            retrieved_ids=retrieved_for_metrics,
            relevant_ids=relevant_for_metrics,
            k=body.k,
        )

        results.append(
            SingleEvalResult(
                query=query,
                metrics=metrics,
                retrieved_questions=retrieved_questions,
                relevant_questions=list(relevant_qs),
            )
        )

        all_query_results.append((retrieved_for_metrics, relevant_for_metrics))

    mrr = mean_reciprocal_rank(all_query_results)
    avg_precision = sum(r.metrics[f"precision@{body.k}"] for r in results) / len(
        results
    )
    avg_recall = sum(r.metrics[f"recall@{body.k}"] for r in results) / len(results)

    aggregate = {
        "mrr": round(mrr, 4),
        f"avg_precision@{body.k}": round(avg_precision, 4),
        f"avg_recall@{body.k}": round(avg_recall, 4),
        "num_queries": len(results),
    }

    logger.info("retrieval_eval_complete", aggregate=aggregate)
    return RetrievalEvalResponse(results=results, aggregate=aggregate)


@router.post("/judge", response_model=JudgeResponse)
@limiter.limit(settings.rate_limit_eval)
async def judge_single(
    request: Request,
    body: JudgeRequest,
    _api_key: str | None = Depends(verify_api_key),
):
    """Evaluate a single question using LLM-as-judge."""
    logger.info("judge_single_start", question=body.question[:80])

    rag_result = generate_answer(body.question)

    context = "\n\n".join(
        s.get("question", "") for s in rag_result.sources
    )

    evaluation = evaluate_response(
        question=body.question,
        context=rag_result.answer,
        answer=rag_result.answer,
    )

    return JudgeResponse(
        question=body.question,
        answer=rag_result.answer,
        evaluation=evaluation,
        sources=rag_result.sources,
    )


@router.post("/full", response_model=FullEvalResponse)
@limiter.limit(settings.rate_limit_eval)
async def full_evaluation(
    request: Request,
    _api_key: str | None = Depends(verify_api_key),
):
    """Run full evaluation: retrieval metrics + LLM-as-judge on all test queries."""
    logger.info("full_eval_start")

    retrieval_result = await evaluate_retrieval_endpoint(
        request, EvalRetrievalRequest(k=5)
    )

    judge_results = []
    for item in EVAL_DATASET[:5]:
        try:
            rag_result = generate_answer(item["query"])
            context_text = "\n\n".join(
                s.get("question", "") for s in rag_result.sources
            )

            evaluation = evaluate_response(
                question=item["query"],
                context=context_text,
                answer=rag_result.answer,
            )
            judge_results.append(
                {
                    "query": item["query"],
                    "answer": rag_result.answer[:300],
                    "evaluation": evaluation,
                }
            )
        except Exception as e:
            logger.error("judge_eval_error", query=item["query"], error=str(e))
            judge_results.append(
                {"query": item["query"], "error": str(e)}
            )

    scored = [
        j["evaluation"]["overall_score"]
        for j in judge_results
        if "evaluation" in j and isinstance(j["evaluation"].get("overall_score"), (int, float))
    ]
    avg_judge_score = round(sum(scored) / len(scored), 2) if scored else 0

    overall_summary = {
        "retrieval": retrieval_result.aggregate,
        "avg_judge_score": avg_judge_score,
        "judge_evaluations_count": len(judge_results),
    }

    logger.info("full_eval_complete", summary=overall_summary)
    return FullEvalResponse(
        retrieval_metrics=retrieval_result.aggregate,
        judge_evaluations=judge_results,
        overall_summary=overall_summary,
    )
