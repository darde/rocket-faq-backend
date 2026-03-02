from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    pinecone_api_key: str
    openrouter_api_key: str
    pinecone_index_name: str = "rocket-mortgage-faq"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_provider: str = "local"  # "local" (sentence-transformers) or "api" (HF Inference API)
    llm_model: str = "google/gemini-2.0-flash-001"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    hf_token: Optional[str] = None
    frontend_url: str = "https://rocket-faq-frontend.vercel.app"
    local_frontend_url: Optional[str] = None
    top_k: int = 5
    log_level: str = "INFO"

    # Rate limiting
    rate_limit_default: str = "60/minute"
    rate_limit_chat: str = "20/minute"
    rate_limit_eval: str = "5/minute"

    # Caching
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 200
    cache_embedding_max_size: int = 500

    # Security
    api_key: Optional[str] = None
    max_question_length: int = 1000

    # Cost optimization
    daily_token_budget: int = 0
    monthly_token_budget: int = 0
    max_response_tokens: int = 1024

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
