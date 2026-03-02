"""In-memory TTL+LRU cache for RAG responses and embeddings."""

from __future__ import annotations

import hashlib
import threading

from cachetools import TTLCache

from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)

settings = get_settings()

_rag_cache: TTLCache = TTLCache(
    maxsize=settings.cache_max_size,
    ttl=settings.cache_ttl_seconds,
)
_rag_lock = threading.Lock()

_embedding_cache: TTLCache = TTLCache(
    maxsize=settings.cache_embedding_max_size,
    ttl=settings.cache_ttl_seconds,
)
_embedding_lock = threading.Lock()


def _normalize_question(question: str) -> str:
    """Lowercase, strip whitespace and trailing punctuation for cache key."""
    return question.strip().lower().rstrip("?!. ")


def _cache_key(text: str) -> str:
    """Generate a stable hash key for a text string."""
    normalized = _normalize_question(text)
    return hashlib.sha256(normalized.encode()).hexdigest()


def get_cached_rag_response(question: str):
    """Return cached RAGResponse or None."""
    key = _cache_key(question)
    with _rag_lock:
        result = _rag_cache.get(key)
    if result is not None:
        logger.info("cache_hit", cache="rag", question=question[:80])
    else:
        logger.info("cache_miss", cache="rag", question=question[:80])
    return result


def set_cached_rag_response(question: str, response) -> None:
    """Store a RAGResponse in cache."""
    key = _cache_key(question)
    with _rag_lock:
        _rag_cache[key] = response
    logger.info("cache_set", cache="rag", question=question[:80])


def get_cached_embedding(text: str) -> list[float] | None:
    """Return cached embedding vector or None."""
    key = _cache_key(text)
    with _embedding_lock:
        return _embedding_cache.get(key)


def set_cached_embedding(text: str, vector: list[float]) -> None:
    """Store an embedding vector in cache."""
    key = _cache_key(text)
    with _embedding_lock:
        _embedding_cache[key] = vector


def get_cache_stats() -> dict:
    """Return current cache statistics for observability."""
    with _rag_lock:
        rag_size = len(_rag_cache)
        rag_max = _rag_cache.maxsize
    with _embedding_lock:
        emb_size = len(_embedding_cache)
        emb_max = _embedding_cache.maxsize
    return {
        "rag_cache_entries": rag_size,
        "rag_cache_max": rag_max,
        "embedding_cache_entries": emb_size,
        "embedding_cache_max": emb_max,
    }
