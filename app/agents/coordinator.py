"""Agent Coordinator — orchestrates all agents and produces combined reports."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from app.agents.code_reviewer import CodeReviewAgent
from app.agents.documenter import DocumenterAgent
from app.agents.tech_debt import TechDebtAgent
from app.agents.models import AgentReport, CoordinatorReport, Severity
from app.config import get_settings
from app.core.llm import BudgetExceededError
from app.observability.logger import get_logger

logger = get_logger(__name__)


class AgentCoordinator:
    """Orchestrates all analysis agents and produces a combined report."""

    def __init__(self):
        self.agents = [
            CodeReviewAgent(),
            TechDebtAgent(),
            DocumenterAgent(),
        ]

    def run_all(self) -> CoordinatorReport:
        """Run all agents sequentially and produce a combined report."""
        start = time.time()
        logger.info("coordinator_start", agents=[a.name for a in self.agents])

        combined = CoordinatorReport()
        agent_reports: list[AgentReport] = []

        for agent in self.agents:
            logger.info("coordinator_running_agent", agent=agent.name)
            try:
                report = agent.analyze()
                agent_reports.append(report)
                combined.agent_reports.append(report.to_dict())
            except BudgetExceededError:
                logger.warning("coordinator_budget_stop", agent=agent.name)
                combined.metadata["stopped_at_agent"] = agent.name
                combined.metadata["budget_exceeded"] = True
                break
            except Exception as e:
                logger.error(
                    "coordinator_agent_error", agent=agent.name, error=str(e)
                )
                combined.agent_reports.append(
                    {"agent_name": agent.name, "error": str(e), "findings": []}
                )

        all_findings = []
        for report in agent_reports:
            all_findings.extend(report.findings)

        combined.total_findings = len(all_findings)
        combined.findings_by_severity = dict(
            Counter(f.severity.value for f in all_findings)
        )

        all_recs = []
        for report in agent_reports:
            all_recs.extend(report.recommendations[:3])
        combined.top_recommendations = all_recs[:10]

        combined.executive_summary = self._generate_executive_summary(
            agent_reports
        )

        combined.metadata["duration_seconds"] = round(time.time() - start, 2)
        combined.metadata["agents_completed"] = len(agent_reports)
        combined.metadata["agents_total"] = len(self.agents)

        logger.info(
            "coordinator_complete",
            total_findings=combined.total_findings,
            duration=combined.metadata["duration_seconds"],
        )
        return combined

    def run_single(self, agent_name: str) -> AgentReport:
        """Run a single agent by name."""
        for agent in self.agents:
            if agent.name == agent_name:
                return agent.analyze()
        raise ValueError(
            f"Unknown agent: {agent_name}. "
            f"Available: {[a.name for a in self.agents]}"
        )

    def _generate_executive_summary(
        self, reports: list[AgentReport]
    ) -> str:
        parts = []
        for report in reports:
            parts.append(f"**{report.agent_name}**: {report.summary}")

        total_findings = sum(len(r.findings) for r in reports)
        high_critical = sum(
            1
            for r in reports
            for f in r.findings
            if f.severity in (Severity.HIGH, Severity.CRITICAL)
        )

        return (
            f"Multi-agent analysis completed. {len(reports)} agents analyzed the codebase, "
            f"producing {total_findings} total findings "
            f"({high_critical} high/critical priority).\n\n"
            + "\n\n".join(parts)
        )

    def save_report(self, report: CoordinatorReport) -> Path:
        """Save the combined coordinator report."""
        settings = get_settings()
        report_dir = Path(settings.agent_report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = report.timestamp.replace(":", "-").replace("+", "_")
        filename = f"full_report_{timestamp}.json"
        path = report_dir / filename

        report_dict = {
            "timestamp": report.timestamp,
            "executive_summary": report.executive_summary,
            "total_findings": report.total_findings,
            "findings_by_severity": report.findings_by_severity,
            "top_recommendations": report.top_recommendations,
            "agent_reports": report.agent_reports,
            "metadata": report.metadata,
        }

        path.write_text(json.dumps(report_dict, indent=2, default=str))
        logger.info("coordinator_report_saved", path=str(path))

        for agent_report_dict in report.agent_reports:
            agent_name = agent_report_dict.get("agent_name", "unknown")
            agent_file = report_dir / f"{agent_name}_{timestamp}.json"
            agent_file.write_text(
                json.dumps(agent_report_dict, indent=2, default=str)
            )

        for ar in report.agent_reports:
            md = ar.get("metadata", {}).get("markdown_doc")
            if md:
                md_path = report_dir / f"documentation_{timestamp}.md"
                md_path.write_text(md)
                logger.info("markdown_doc_saved", path=str(md_path))

        return path
