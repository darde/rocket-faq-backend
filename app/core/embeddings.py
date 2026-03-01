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
    """HuggingFace Inference Providers API — zero ML dependencies."""

    def __init__(self):
        from huggingface_hub import InferenceClient

        settings = get_settings()
        model_id = settings.embedding_model
        if not model_id.startswith("sentence-transformers/"):
            model_id = f"sentence-transformers/{model_id}"
        self._model_id = model_id
        self._client = InferenceClient(
            token=settings.hf_token,
            timeout=30,
        )
        logger.info("loading_api_embeddings", model=model_id)

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0 else vec

    def _embed(self, text: str) -> list[float]:
        result = self._client.feature_extraction(
            text, model=self._model_id, normalize=True
        )
        vec = result.tolist() if hasattr(result, "tolist") else result
        if vec and isinstance(vec[0], list):
            vec = vec[0]
        return self._normalize(vec)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


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
