from __future__ import annotations

import time
from pinecone import Pinecone, ServerlessSpec

from app.config import get_settings
from app.core.chunking import Chunk
from app.core.embeddings import get_embeddings_provider
from app.observability.logger import get_logger

logger = get_logger(__name__)

_pinecone_index = None


def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        settings = get_settings()
        pc = Pinecone(api_key=settings.pinecone_api_key)
        _pinecone_index = pc.Index(settings.pinecone_index_name)
        logger.info("pinecone_connected", index=settings.pinecone_index_name)
    return _pinecone_index


def ensure_index_exists():
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if settings.pinecone_index_name in existing:
        logger.info("index_exists", index=settings.pinecone_index_name)
        return

    embeddings = get_embeddings_provider()
    sample = embeddings.embed_query("test")
    dimension = len(sample)

    logger.info(
        "creating_index",
        index=settings.pinecone_index_name,
        dimension=dimension,
    )
    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    while not pc.describe_index(settings.pinecone_index_name).status.get("ready"):
        time.sleep(1)
    logger.info("index_ready", index=settings.pinecone_index_name)


def upsert_chunks(chunks: list[Chunk], batch_size: int = 50):
    embeddings = get_embeddings_provider()
    index = get_pinecone_index()

    texts = [c.text for c in chunks]
    logger.info("generating_embeddings", count=len(texts))
    vectors = embeddings.embed_documents(texts)

    records = []
    for chunk, vector in zip(chunks, vectors):
        records.append(
            {
                "id": chunk.id,
                "values": vector,
                "metadata": {**chunk.metadata, "text": chunk.text},
            }
        )

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        index.upsert(vectors=batch)
        logger.info("upserted_batch", start=i, size=len(batch))

    logger.info("upsert_complete", total=len(records))


def search(query: str, top_k: int | None = None) -> list[dict]:
    settings = get_settings()
    if top_k is None:
        top_k = settings.top_k

    embeddings = get_embeddings_provider()
    index = get_pinecone_index()

    query_vector = embeddings.embed_query(query)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    docs = []
    for match in results.matches:
        docs.append(
            {
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "metadata": {
                    k: v for k, v in match.metadata.items() if k != "text"
                },
            }
        )

    logger.info("search_complete", query=query[:80], top_k=top_k, results=len(docs))
    return docs
