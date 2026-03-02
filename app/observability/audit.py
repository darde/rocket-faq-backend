"""JSONL audit log for Q&A interactions and governance reporting."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)

_audit_lock = threading.Lock()


@dataclass
class AuditEntry:
    timestamp: str
    request_id: str
    question_redacted: str
    question_had_pii: bool = False
    pii_types_detected: list[str] = field(default_factory=list)
    injection_detected: bool = False
    injection_risk_level: str = "none"
    off_topic: bool = False
    answer: str = ""
    answer_had_pii: bool = False
    disclaimers_added: list[str] = field(default_factory=list)
    low_confidence: bool = False
    sources: list[dict] = field(default_factory=list)
    blocked: bool = False
    blocked_reason: str | None = None
    feedback: dict | None = None


def _get_log_path() -> Path:
    settings = get_settings()
    return Path(settings.audit_log_path)


def write_audit_entry(entry: AuditEntry) -> None:
    """Append a single audit entry as a JSON line."""
    settings = get_settings()
    if not settings.audit_log_enabled:
        return

    path = _get_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with _audit_lock:
        with open(path, "a") as f:
            f.write(json.dumps(asdict(entry), default=str) + "\n")

    logger.info(
        "audit_entry_written",
        request_id=entry.request_id,
        blocked=entry.blocked,
        pii=entry.question_had_pii,
        injection=entry.injection_detected,
    )


def update_audit_feedback(
    request_id: str, rating: str, comment: str | None = None
) -> bool:
    """Find and update the feedback field for a given request_id."""
    path = _get_log_path()
    if not path.exists():
        return False

    with _audit_lock:
        lines = path.read_text().strip().split("\n")
        updated = False
        new_lines = []

        for line in lines:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("request_id") == request_id and entry.get("feedback") is None:
                entry["feedback"] = {"rating": rating, "comment": comment}
                updated = True
            new_lines.append(json.dumps(entry, default=str))

        if updated:
            path.write_text("\n".join(new_lines) + "\n")

    if updated:
        logger.info("audit_feedback_updated", request_id=request_id, rating=rating)

    return updated


def get_governance_summary() -> dict:
    """Aggregate stats from audit log for the governance endpoint."""
    path = _get_log_path()

    summary = {
        "total_queries": 0,
        "flagged_queries": {
            "pii_detected": 0,
            "injection_detected": 0,
            "off_topic": 0,
            "blocked": 0,
        },
        "low_confidence_count": 0,
        "disclaimers_count": {
            "financial_advice": 0,
            "legal": 0,
            "tax": 0,
        },
        "feedback_summary": {
            "positive": 0,
            "negative": 0,
            "total": 0,
        },
        "avg_source_score": None,
    }

    if not path.exists():
        return summary

    total_scores = []

    with _audit_lock:
        for line in path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            summary["total_queries"] += 1

            if entry.get("question_had_pii"):
                summary["flagged_queries"]["pii_detected"] += 1
            if entry.get("injection_detected"):
                summary["flagged_queries"]["injection_detected"] += 1
            if entry.get("off_topic"):
                summary["flagged_queries"]["off_topic"] += 1
            if entry.get("blocked"):
                summary["flagged_queries"]["blocked"] += 1
            if entry.get("low_confidence"):
                summary["low_confidence_count"] += 1

            for disclaimer in entry.get("disclaimers_added", []):
                if disclaimer in summary["disclaimers_count"]:
                    summary["disclaimers_count"][disclaimer] += 1

            feedback = entry.get("feedback")
            if feedback and isinstance(feedback, dict):
                rating = feedback.get("rating")
                if rating == "positive":
                    summary["feedback_summary"]["positive"] += 1
                elif rating == "negative":
                    summary["feedback_summary"]["negative"] += 1
                summary["feedback_summary"]["total"] += 1

            for source in entry.get("sources", []):
                if isinstance(source, dict) and "score" in source:
                    total_scores.append(source["score"])

    if total_scores:
        summary["avg_source_score"] = round(sum(total_scores) / len(total_scores), 4)

    return summary
