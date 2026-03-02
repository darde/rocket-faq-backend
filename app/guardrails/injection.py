"""Prompt injection detection using regex pattern matching."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "instruction_override",
        re.compile(
            r"(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:previous|above|prior|earlier|your)\s+(?:instructions|rules|prompts|guidelines|directives)",
            re.IGNORECASE,
        ),
    ),
    (
        "role_reassignment",
        re.compile(
            r"(?:you are now|act as|pretend (?:to be|you're)|from now on you|assume the (?:role|identity)|new role|new persona)",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_extraction",
        re.compile(
            r"(?:(?:what|show|reveal|repeat|print|display|output|tell me)\s+(?:your|the|system)\s+(?:system\s+)?(?:prompt|instructions|rules|directives))|(?:system\s*prompt)",
            re.IGNORECASE,
        ),
    ),
    (
        "delimiter_injection",
        re.compile(
            r"(?:---+|===+|###)\s*(?:system|instruction|new prompt|end|begin)",
            re.IGNORECASE,
        ),
    ),
    (
        "code_execution",
        re.compile(
            r"(?:base64|eval\s*\(|exec\s*\(|import\s+os|subprocess|__import__)",
            re.IGNORECASE,
        ),
    ),
    (
        "jailbreak",
        re.compile(
            r"\b(?:DAN|do anything now|jailbreak|developer mode|DEVELOPER MODE|unfiltered mode)\b",
            re.IGNORECASE,
        ),
    ),
]


@dataclass
class InjectionScanResult:
    is_injection: bool
    risk_level: str  # "none", "low", "high"
    matched_patterns: list[str] = field(default_factory=list)
    explanation: str = ""


def scan_injection(text: str) -> InjectionScanResult:
    """Scan user input for prompt injection patterns."""
    matched: list[str] = []

    for name, pattern in _PATTERNS:
        if pattern.search(text):
            matched.append(name)

    if not matched:
        return InjectionScanResult(
            is_injection=False,
            risk_level="none",
            matched_patterns=[],
            explanation="",
        )

    risk_level = "high" if len(matched) >= 2 else "low"
    explanation = (
        f"Detected {len(matched)} injection pattern(s): {', '.join(matched)}. "
        f"Risk level: {risk_level}."
    )

    return InjectionScanResult(
        is_injection=True,
        risk_level=risk_level,
        matched_patterns=matched,
        explanation=explanation,
    )
