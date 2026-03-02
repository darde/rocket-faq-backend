from __future__ import annotations

from openai import OpenAI

from app.config import get_settings
from app.observability.logger import get_logger
from app.observability.cost_tracker import usage_tracker

logger = get_logger(__name__)

_client: OpenAI | None = None


class BudgetExceededError(Exception):
    """Raised when the token budget has been exhausted."""

    pass


def get_openrouter_client() -> OpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )
        logger.info("openrouter_client_initialized")
    return _client


def chat_completion(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> str:
    settings = get_settings()
    client = get_openrouter_client()
    model = model or settings.llm_model

    if max_tokens is None:
        max_tokens = settings.max_response_tokens

    within_budget, budget_msg = usage_tracker.check_budget()
    if not within_budget:
        logger.warning("budget_exceeded", message=budget_msg)
        raise BudgetExceededError(budget_msg)

    logger.info(
        "llm_request",
        model=model,
        messages_count=len(messages),
        temperature=temperature,
        max_tokens=max_tokens,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content or ""

    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0

    usage_info = usage_tracker.record_usage(model, prompt_tokens, completion_tokens)

    logger.info(
        "llm_response",
        model=model,
        usage_prompt=prompt_tokens,
        usage_completion=completion_tokens,
        response_length=len(content),
        cost_usd=usage_info["request_cost_usd"],
        daily_cost_usd=usage_info["daily_cost_usd"],
    )
    return content
