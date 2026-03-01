from __future__ import annotations

from dataclasses import dataclass

from app.core.vectorstore import search
from app.core.llm import chat_completion
from app.observability.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the Rocket Mortgage FAQ Assistant, a helpful and knowledgeable \
customer support bot. Your job is to answer questions about Rocket Mortgage services, \
loan management, payments, escrow, taxes, insurance, and related mortgage topics.

Rules:
1. Use the provided context from the knowledge base to answer questions accurately.
2. If the context contains the answer, respond based on that information.
3. For general mortgage questions not specific to Rocket Mortgage, you may use your general knowledge, \
   but clearly indicate when you're providing general information.
4. If you don't have enough information to answer, say so honestly and suggest the user \
   contact Rocket Mortgage directly.
5. Be concise, professional, and friendly.
6. When referencing contact information, always provide the specific numbers/emails from the context."""

CONTEXT_TEMPLATE = """Here is relevant information from the Rocket Mortgage knowledge base:

{context}

---
User question: {question}"""


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    query: str


def generate_answer(query: str, top_k: int | None = None) -> RAGResponse:
    """Full RAG pipeline: retrieve relevant chunks, then generate an answer."""
    logger.info("rag_pipeline_start", query=query[:100])

    retrieved_docs = search(query, top_k=top_k)

    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        section = doc["metadata"].get("section", "")
        subsection = doc["metadata"].get("subsection", "")
        source_label = f"[{section}"
        if subsection:
            source_label += f" > {subsection}"
        source_label += f"] (relevance: {doc['score']:.3f})"
        context_parts.append(f"--- Source {i} {source_label} ---\n{doc['text']}")

    context = "\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": CONTEXT_TEMPLATE.format(context=context, question=query),
        },
    ]

    answer = chat_completion(messages)

    sources = [
        {
            "id": doc["id"],
            "score": doc["score"],
            "section": doc["metadata"].get("section", ""),
            "subsection": doc["metadata"].get("subsection", ""),
            "question": doc["metadata"].get("question", ""),
        }
        for doc in retrieved_docs
    ]

    logger.info("rag_pipeline_complete", query=query[:100], sources_used=len(sources))
    return RAGResponse(answer=answer, sources=sources, query=query)
