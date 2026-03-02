"""Output guardrails: disclaimers, PII leak check, confidence flagging."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.guardrails.pii import scan_pii

DISCLAIMER_KEYWORDS: dict[str, list[str]] = {
    "financial_advice": [
        "financial advice", "invest", "recommend buying", "you should buy",
        "guaranteed return", "profit", "financial planning",
    ],
    "legal": [
        "legal advice", "attorney", "lawsuit", "sue", "court", "legally",
        "legal action", "lawyer",
    ],
    "tax": [
        "tax advice", "tax deduction", "write off", "tax return", "irs",
        "1099", "tax professional", "tax liability",
    ],
}

DISCLAIMERS: dict[str, str] = {
    "financial_advice": (
        "\n\n---\n*Disclaimer: This information is for educational purposes only "
        "and should not be considered financial advice. Please consult a qualified "
        "financial advisor for personalized guidance.*"
    ),
    "legal": (
        "\n\n---\n*Disclaimer: This is general information, not legal advice. "
        "Please consult a qualified attorney for legal matters.*"
    ),
    "tax": (
        "\n\n---\n*Disclaimer: This is general information, not tax advice. "
        "Please consult a qualified tax professional for specific tax questions.*"
    ),
}

LOW_CONFIDENCE_WARNING = (
    "*Note: I could not find highly relevant information in the knowledge base "
    "for this question. The following response may be less reliable. Consider "
    "contacting Rocket Mortgage directly for accurate information.*\n\n"
)

# Build compiled patterns for disclaimer detection
_DISCLAIMER_PATTERNS: dict[str, re.Pattern] = {
    category: re.compile(
        r"\b(?:" + "|".join(re.escape(kw) for kw in keywords) + r")\b",
        re.IGNORECASE,
    )
    for category, keywords in DISCLAIMER_KEYWORDS.items()
}


@dataclass
class OutputGuardrailResult:
    modified_answer: str
    disclaimers_added: list[str] = field(default_factory=list)
    pii_leaked: bool = False
    low_confidence: bool = False


def process_output(
    answer: str,
    source_scores: list[float],
    confidence_threshold: float = 0.5,
) -> OutputGuardrailResult:
    """Apply all output guardrails to the LLM response."""
    disclaimers_added: list[str] = []
    modified = answer

    # 1. Check for PII leaks in the output
    pii_result = scan_pii(answer)
    pii_leaked = pii_result.has_pii
    if pii_leaked:
        modified = pii_result.redacted_text

    # 2. Low confidence check
    low_confidence = False
    if source_scores and all(score < confidence_threshold for score in source_scores):
        low_confidence = True
        modified = LOW_CONFIDENCE_WARNING + modified

    # 3. Disclaimer injection (at most one, prioritized)
    for category in ["financial_advice", "legal", "tax"]:
        if _DISCLAIMER_PATTERNS[category].search(modified):
            modified += DISCLAIMERS[category]
            disclaimers_added.append(category)
            break  # Only one disclaimer

    return OutputGuardrailResult(
        modified_answer=modified,
        disclaimers_added=disclaimers_added,
        pii_leaked=pii_leaked,
        low_confidence=low_confidence,
    )
