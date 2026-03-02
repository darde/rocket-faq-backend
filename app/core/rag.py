from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.config import get_settings
from app.core.vectorstore import search
from app.core.llm import chat_completion
from app.core.cache import get_cached_rag_response, set_cached_rag_response
from app.guardrails.pii import scan_pii
from app.guardrails.injection import scan_injection
from app.guardrails.topic import scan_topic
from app.guardrails.output import process_output
from app.observability.audit import AuditEntry, write_audit_entry
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
6. When referencing contact information, always provide the specific numbers/emails from the context.

Responsible AI Guidelines:
7. NEVER provide specific financial, legal, or tax advice. You may share general educational \
   information but always recommend consulting qualified professionals for personalized advice.
8. NEVER ask for or encourage users to share sensitive personal information such as Social Security \
   numbers, credit card numbers, bank account numbers, or passwords. If a user shares such \
   information, do not repeat it and remind them to avoid sharing sensitive data in chat.
9. If a user's question is unrelated to mortgages or Rocket Mortgage services, politely redirect \
   them: explain that you specialize in mortgage-related topics and suggest appropriate resources.
10. If a user appears to be in financial distress, be empathetic and suggest contacting Rocket \
    Mortgage's hardship team or a HUD-approved housing counselor.
11. Never generate content that is discriminatory, harmful, or violates fair lending practices.
12. Acknowledge your limitations: you are an AI assistant with access to a knowledge base, \
    not a licensed mortgage professional."""

CONTEXT_TEMPLATE = """Here is relevant information from the Rocket Mortgage knowledge base:

{context}

---
User question: {question}"""


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    query: str
    guardrails: dict = field(default_factory=dict)


def _build_guardrails_meta() -> dict:
    return {
        "input_pii_detected": False,
        "input_pii_types": [],
        "injection_detected": False,
        "injection_risk_level": "none",
        "off_topic": False,
        "output_pii_leaked": False,
        "disclaimers_added": [],
        "low_confidence": False,
        "blocked": False,
        "blocked_reason": None,
    }


def _write_audit(
    request_id: str,
    query: str,
    pii_redacted_query: str,
    guardrails_meta: dict,
    answer: str,
    sources: list[dict],
) -> None:
    entry = AuditEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        request_id=request_id,
        question_redacted=pii_redacted_query,
        question_had_pii=guardrails_meta["input_pii_detected"],
        pii_types_detected=guardrails_meta["input_pii_types"],
        injection_detected=guardrails_meta["injection_detected"],
        injection_risk_level=guardrails_meta["injection_risk_level"],
        off_topic=guardrails_meta["off_topic"],
        answer=answer,
        answer_had_pii=guardrails_meta["output_pii_leaked"],
        disclaimers_added=guardrails_meta["disclaimers_added"],
        low_confidence=guardrails_meta["low_confidence"],
        sources=[{"id": s.get("id", ""), "score": s.get("score", 0)} for s in sources],
        blocked=guardrails_meta["blocked"],
        blocked_reason=guardrails_meta["blocked_reason"],
    )
    write_audit_entry(entry)


def generate_answer(
    query: str, top_k: int | None = None, request_id: str = ""
) -> RAGResponse:
    """Full RAG pipeline with input/output guardrails."""
    settings = get_settings()
    logger.info("rag_pipeline_start", query=query[:100])

    guardrails_meta = _build_guardrails_meta()

    # --- INPUT GUARDRAILS ---
    pii_redacted_query = query

    if settings.pii_detection_enabled:
        pii_result = scan_pii(query)
        guardrails_meta["input_pii_detected"] = pii_result.has_pii
        guardrails_meta["input_pii_types"] = [m.type for m in pii_result.matches]
        pii_redacted_query = pii_result.redacted_text
        if pii_result.has_pii:
            logger.warning(
                "pii_detected_in_input",
                types=[m.type for m in pii_result.matches],
            )

    if settings.injection_detection_enabled:
        injection_result = scan_injection(query)
        guardrails_meta["injection_detected"] = injection_result.is_injection
        guardrails_meta["injection_risk_level"] = injection_result.risk_level
        if injection_result.risk_level == "high":
            guardrails_meta["blocked"] = True
            guardrails_meta["blocked_reason"] = "prompt_injection"
            logger.warning("prompt_injection_blocked", patterns=injection_result.matched_patterns)
            answer = (
                "I'm sorry, but I can't process that request. If you have a question "
                "about Rocket Mortgage, please rephrase it."
            )
            response = RAGResponse(
                answer=answer, sources=[], query=query, guardrails=guardrails_meta
            )
            _write_audit(request_id, query, pii_redacted_query, guardrails_meta, answer, [])
            return response
        elif injection_result.is_injection:
            logger.warning(
                "prompt_injection_warning",
                risk_level=injection_result.risk_level,
                patterns=injection_result.matched_patterns,
            )

    if settings.topic_detection_enabled:
        topic_result = scan_topic(query)
        guardrails_meta["off_topic"] = topic_result.is_off_topic
        if topic_result.is_off_topic:
            guardrails_meta["blocked"] = True
            guardrails_meta["blocked_reason"] = "off_topic"
            logger.info("off_topic_detected", query=query[:100])
            answer = (
                "I appreciate your question, but I'm specifically designed to help "
                "with Rocket Mortgage and mortgage-related topics. For other inquiries, "
                "please reach out to the appropriate service. Is there anything about "
                "your mortgage I can help with?"
            )
            response = RAGResponse(
                answer=answer, sources=[], query=query, guardrails=guardrails_meta
            )
            _write_audit(request_id, query, pii_redacted_query, guardrails_meta, answer, [])
            return response

    # --- CACHE CHECK (use redacted query) ---
    if top_k is None:
        cached = get_cached_rag_response(pii_redacted_query)
        if cached is not None:
            _write_audit(
                request_id, query, pii_redacted_query, guardrails_meta,
                cached.answer, cached.sources,
            )
            return RAGResponse(
                answer=cached.answer,
                sources=cached.sources,
                query=query,
                guardrails=guardrails_meta,
            )

    # --- RAG PIPELINE ---
    retrieved_docs = search(pii_redacted_query, top_k=top_k)

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
            "content": CONTEXT_TEMPLATE.format(
                context=context, question=pii_redacted_query
            ),
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

    # --- OUTPUT GUARDRAILS ---
    source_scores = [doc["score"] for doc in retrieved_docs]
    output_result = process_output(
        answer, source_scores, settings.confidence_threshold
    )

    guardrails_meta["output_pii_leaked"] = output_result.pii_leaked
    guardrails_meta["disclaimers_added"] = output_result.disclaimers_added
    guardrails_meta["low_confidence"] = output_result.low_confidence

    final_answer = output_result.modified_answer

    # --- CACHE STORE ---
    cache_response = RAGResponse(answer=final_answer, sources=sources, query=pii_redacted_query)
    if top_k is None:
        set_cached_rag_response(pii_redacted_query, cache_response)

    # --- AUDIT ---
    _write_audit(request_id, query, pii_redacted_query, guardrails_meta, final_answer, sources)

    logger.info("rag_pipeline_complete", query=query[:100], sources_used=len(sources))
    return RAGResponse(
        answer=final_answer, sources=sources, query=query, guardrails=guardrails_meta
    )
