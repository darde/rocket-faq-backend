"""Ingest source.md into Pinecone vector database."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.core.chunking import chunk_faq_document
from app.core.vectorstore import ensure_index_exists, upsert_chunks
from app.observability.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def main():
    settings = get_settings()
    source_path = Path(__file__).resolve().parent.parent / "source.md"

    if not source_path.exists():
        logger.error("source_file_not_found", path=str(source_path))
        sys.exit(1)

    logger.info("reading_source", path=str(source_path))
    content = source_path.read_text(encoding="utf-8")

    logger.info("chunking_document")
    chunks = chunk_faq_document(content)
    logger.info("chunks_created", count=len(chunks))

    for i, chunk in enumerate(chunks[:3]):
        logger.info(
            "sample_chunk",
            index=i,
            section=chunk.metadata.get("section"),
            question=chunk.metadata.get("question", "")[:80],
            text_length=len(chunk.text),
        )

    logger.info("ensuring_pinecone_index", index=settings.pinecone_index_name)
    ensure_index_exists()

    logger.info("upserting_chunks")
    upsert_chunks(chunks)

    logger.info(
        "ingestion_complete",
        total_chunks=len(chunks),
        index=settings.pinecone_index_name,
    )


if __name__ == "__main__":
    main()
