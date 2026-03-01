from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    pinecone_api_key: str
    openrouter_api_key: str
    pinecone_index_name: str = "rocket-mortgage-faq"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_provider: str = "local"  # "local" (sentence-transformers) or "api" (HF Inference API)
    llm_model: str = "google/gemini-2.0-flash-001"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    frontend_url: str = "http://localhost:5173"
    top_k: int = 5
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
