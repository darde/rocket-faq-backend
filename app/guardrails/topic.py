"""Off-topic detection using keyword matching."""

from __future__ import annotations

import re
from dataclasses import dataclass

MORTGAGE_KEYWORDS = {
    "mortgage", "loan", "payment", "escrow", "pmi", "mip", "refinance", "refi",
    "interest rate", "closing", "down payment", "appraisal", "homeowner", "property",
    "insurance", "tax", "1098", "heloc", "credit", "forbearance", "autopay",
    "rocket mortgage", "quicken", "title", "deed", "principal", "amortization",
    "preapproval", "pre-approval", "underwriting", "lender", "borrower",
    "monthly payment", "arm", "fixed rate", "equity", "lien", "foreclosure",
    "default", "delinquent", "payoff", "account", "servicing", "statement",
    "balance", "billing", "home equity", "house", "real estate", "closing costs",
    "origination", "points", "rate lock", "conventional", "fha", "va loan",
    "usda", "jumbo", "conforming", "piti", "dti", "ltv",
    "grace period", "late fee", "due date", "flood insurance",
    "hazard insurance", "loan modification",
}

GREETING_PATTERNS = re.compile(
    r"^\s*(?:hi|hello|hey|good (?:morning|afternoon|evening)|thanks|thank you|"
    r"help|who are you|what (?:can you|do you) do|how are you)\s*[?!.]?\s*$",
    re.IGNORECASE,
)

# Build a single regex from all keywords for efficient matching
_KEYWORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in MORTGAGE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


@dataclass
class TopicScanResult:
    is_off_topic: bool
    has_mortgage_keywords: bool
    explanation: str


def scan_topic(question: str) -> TopicScanResult:
    """Check if a question is related to mortgages/Rocket Mortgage."""
    # Allow greetings and meta questions
    if GREETING_PATTERNS.match(question):
        return TopicScanResult(
            is_off_topic=False,
            has_mortgage_keywords=False,
            explanation="Greeting or meta question detected.",
        )

    has_keywords = bool(_KEYWORD_PATTERN.search(question))

    if has_keywords:
        return TopicScanResult(
            is_off_topic=False,
            has_mortgage_keywords=True,
            explanation="",
        )

    return TopicScanResult(
        is_off_topic=True,
        has_mortgage_keywords=False,
        explanation="No mortgage-related keywords found in question.",
    )
