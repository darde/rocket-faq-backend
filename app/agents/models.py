"""Data models for the multi-agent analysis system."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone


class Severity(str, enum.Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Finding:
    """A single finding from an agent analysis."""

    category: str
    severity: Severity
    file_path: str
    description: str
    suggestion: str
    line_range: str = ""


@dataclass
class AgentReport:
    """Complete report from a single agent run."""

    agent_name: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    summary: str = ""
    findings: list[Finding] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "findings": [
                {
                    "category": f.category,
                    "severity": f.severity.value,
                    "file_path": f.file_path,
                    "description": f.description,
                    "suggestion": f.suggestion,
                    "line_range": f.line_range,
                }
                for f in self.findings
            ],
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


@dataclass
class CoordinatorReport:
    """Combined report from all agents."""

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    agent_reports: list[dict] = field(default_factory=list)
    executive_summary: str = ""
    total_findings: int = 0
    findings_by_severity: dict = field(default_factory=dict)
    top_recommendations: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
