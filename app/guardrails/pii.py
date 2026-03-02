"""PII detection and redaction using regex patterns."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Known Rocket Mortgage contact info (do not redact these)
PHONE_ALLOWLIST = {
    "8008634332", "800-863-4332", "(800) 863-4332",
    "8007851085", "800-785-1085", "(800) 785-1085",
    "8005080944", "800-508-0944", "(800) 508-0944",
    "8006462133", "800-646-2133", "(800) 646-2133",
    "8552828722", "855-282-8722", "(855) 282-8722",
    "8669478425", "866-947-8425", "(866) 947-8425",
    "8002793005", "800-279-3005", "(800) 279-3005",
    "8336580524", "833-658-0524", "(833) 658-0524",
}

EMAIL_DOMAIN_ALLOWLIST = {"rocketmortgage.com", "quickenloans.com", "rockmortgage.com"}

# Compiled regex patterns
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_SSN_LOOSE_PATTERN = re.compile(
    r"(?:ssn|social\s*security(?:\s*number)?)\s*(?:is|:)?\s*(\d{9})\b",
    re.IGNORECASE,
)
_CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
_PHONE_PATTERN = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_ACCOUNT_PATTERN = re.compile(
    r"(?:account|loan|acct|#)\s*(?:number|num|no\.?|#)?\s*:?\s*(\d{8,12})\b",
    re.IGNORECASE,
)


@dataclass
class PIIMatch:
    type: str
    original: str
    start: int
    end: int


@dataclass
class PIIScanResult:
    has_pii: bool
    matches: list[PIIMatch] = field(default_factory=list)
    redacted_text: str = ""


def _normalize_phone(phone: str) -> str:
    """Strip formatting from phone number for allowlist comparison."""
    return re.sub(r"[^\d]", "", phone)


def _is_allowed_phone(phone: str) -> bool:
    normalized = _normalize_phone(phone)
    return any(_normalize_phone(allowed) == normalized for allowed in PHONE_ALLOWLIST)


def _is_allowed_email(email: str) -> bool:
    domain = email.split("@")[-1].lower()
    return domain in EMAIL_DOMAIN_ALLOWLIST


def scan_pii(text: str) -> PIIScanResult:
    """Scan text for PII patterns and return matches with redacted version."""
    matches: list[PIIMatch] = []
    redacted = text

    # SSN (formatted)
    for m in _SSN_PATTERN.finditer(text):
        matches.append(PIIMatch("ssn", m.group(), m.start(), m.end()))

    # SSN (loose, with context keyword)
    for m in _SSN_LOOSE_PATTERN.finditer(text):
        matches.append(PIIMatch("ssn", m.group(1), m.start(1), m.end(1)))

    # Credit card
    for m in _CREDIT_CARD_PATTERN.finditer(text):
        digits = re.sub(r"[^\d]", "", m.group())
        if len(digits) >= 13:
            matches.append(PIIMatch("credit_card", m.group(), m.start(), m.end()))

    # Phone (skip allowlisted numbers)
    for m in _PHONE_PATTERN.finditer(text):
        if not _is_allowed_phone(m.group()):
            matches.append(PIIMatch("phone", m.group(), m.start(), m.end()))

    # Email (skip allowlisted domains)
    for m in _EMAIL_PATTERN.finditer(text):
        if not _is_allowed_email(m.group()):
            matches.append(PIIMatch("email", m.group(), m.start(), m.end()))

    # Account numbers (with context keyword)
    for m in _ACCOUNT_PATTERN.finditer(text):
        matches.append(PIIMatch("account_number", m.group(1), m.start(1), m.end(1)))

    # Deduplicate by position and sort by start position descending for safe replacement
    seen_positions: set[tuple[int, int]] = set()
    unique_matches: list[PIIMatch] = []
    for match in matches:
        pos = (match.start, match.end)
        if pos not in seen_positions:
            seen_positions.add(pos)
            unique_matches.append(match)

    # Redact from end to start to preserve positions
    unique_matches.sort(key=lambda m: m.start, reverse=True)
    for match in unique_matches:
        label = {
            "ssn": "[SSN REDACTED]",
            "credit_card": "[CREDIT CARD REDACTED]",
            "phone": "[PHONE REDACTED]",
            "email": "[EMAIL REDACTED]",
            "account_number": "[ACCOUNT NUMBER REDACTED]",
        }.get(match.type, "[PII REDACTED]")
        redacted = redacted[: match.start] + label + redacted[match.end :]

    return PIIScanResult(
        has_pii=len(unique_matches) > 0,
        matches=unique_matches,
        redacted_text=redacted,
    )


def redact_pii(text: str) -> str:
    """Convenience wrapper: return redacted text only."""
    return scan_pii(text).redacted_text
