from __future__ import annotations

from openai import OpenAI

from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)

_client: OpenAI | None = None


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
    max_tokens: int = 1024,
) -> str:
    settings = get_settings()
    client = get_openrouter_client()
    model = model or settings.llm_model

    logger.info(
        "llm_request",
        model=model,
        messages_count=len(messages),
        temperature=temperature,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content or ""
    logger.info(
        "llm_response",
        model=model,
        usage_prompt=response.usage.prompt_tokens if response.usage else 0,
        usage_completion=response.usage.completion_tokens if response.usage else 0,
        response_length=len(content),
    )
    return content
