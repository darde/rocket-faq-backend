"""Embedding providers — switchable between local models and the HuggingFace Inference API.

Set EMBEDDING_PROVIDER=local  for development (uses sentence-transformers + PyTorch).
Set EMBEDDING_PROVIDER=api    for lightweight deploys (HTTP calls only, no ML libs).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)


class EmbeddingsProvider(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbeddings(EmbeddingsProvider):
    """Sentence-transformers via LangChain (requires torch)."""

    def __init__(self):
        from langchain_huggingface import HuggingFaceEmbeddings

        settings = get_settings()
        logger.info("loading_local_embeddings", model=settings.embedding_model)
        self._model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed_documents(texts)


class APIEmbeddings(EmbeddingsProvider):
    """HuggingFace Inference API — zero ML dependencies."""

    HF_API_BASE = "https://api-inference.huggingface.co/pipeline/feature-extraction"

    def __init__(self):
        settings = get_settings()
        model_id = settings.embedding_model
        if not model_id.startswith("sentence-transformers/"):
            model_id = f"sentence-transformers/{model_id}"
        self._url = f"{self.HF_API_BASE}/{model_id}"
        logger.info("loading_api_embeddings", url=self._url)

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0 else vec

    def _post(self, inputs: str | list[str]) -> list:
        import requests

        resp = requests.post(
            self._url,
            json={"inputs": inputs, "options": {"wait_for_model": True}},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _mean_pool(token_embeddings: list[list[float]]) -> list[float]:
        n = len(token_embeddings)
        dim = len(token_embeddings[0])
        return [sum(token_embeddings[t][d] for t in range(n)) / n for d in range(dim)]

    def _to_vector(self, raw) -> list[float]:
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            return self._normalize(self._mean_pool(raw))
        return self._normalize(raw)

    def embed_query(self, text: str) -> list[float]:
        return self._to_vector(self._post(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = self._post(texts)
        return [self._to_vector(r) for r in results]


_provider: EmbeddingsProvider | None = None


def get_embeddings_provider() -> EmbeddingsProvider:
    global _provider
    if _provider is None:
        settings = get_settings()
        if settings.embedding_provider == "api":
            _provider = APIEmbeddings()
        else:
            _provider = LocalEmbeddings()
    return _provider
