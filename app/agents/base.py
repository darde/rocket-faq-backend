"""Base class for all analysis agents."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path

from app.agents.models import AgentReport, Finding
from app.config import get_settings
from app.core.llm import chat_completion, BudgetExceededError
from app.observability.logger import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Abstract base for analysis agents."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def _system_prompt(self) -> str: ...

    @abstractmethod
    def _get_file_patterns(self) -> list[str]: ...

    @abstractmethod
    def _build_analysis_prompt(self, files: dict[str, str]) -> str: ...

    @abstractmethod
    def _parse_findings(self, raw: str, files: dict[str, str]) -> list[Finding]: ...

    def analyze(self) -> AgentReport:
        """Run the full agent analysis pipeline."""
        start = time.time()
        settings = get_settings()
        logger.info("agent_start", agent=self.name)

        report = AgentReport(agent_name=self.name)
        all_files = self._read_project_files(self._get_file_patterns())
        report.metadata["files_analyzed"] = len(all_files)
        report.metadata["file_list"] = list(all_files.keys())

        batch_size = settings.agent_max_files_per_batch
        file_items = list(all_files.items())

        for i in range(0, len(file_items), batch_size):
            batch = dict(file_items[i : i + batch_size])
            logger.info(
                "agent_batch",
                agent=self.name,
                batch=i // batch_size + 1,
                files=list(batch.keys()),
            )
            try:
                findings = self._analyze_batch(batch)
                report.findings.extend(findings)
            except BudgetExceededError:
                logger.warning(
                    "agent_budget_exceeded",
                    agent=self.name,
                    batch=i // batch_size + 1,
                )
                report.metadata["budget_exceeded"] = True
                break

        if report.findings:
            try:
                summary, recommendations = self._generate_summary(report.findings)
                report.summary = summary
                report.recommendations = recommendations
            except BudgetExceededError:
                report.summary = (
                    f"Analysis found {len(report.findings)} findings "
                    f"(summary skipped due to budget)."
                )

        report.metadata["duration_seconds"] = round(time.time() - start, 2)
        logger.info(
            "agent_complete",
            agent=self.name,
            findings=len(report.findings),
            duration=report.metadata["duration_seconds"],
        )
        return report

    def _read_project_files(self, patterns: list[str]) -> dict[str, str]:
        """Read files matching glob patterns from the project root."""
        backend_root = Path(__file__).resolve().parent.parent.parent
        project_root = backend_root.parent

        files: dict[str, str] = {}
        exclude = {"venv", "__pycache__", "node_modules", ".git", "dist"}

        for pattern in patterns:
            for path in project_root.glob(pattern):
                if path.is_file() and not any(
                    part in exclude for part in path.parts
                ):
                    try:
                        rel = str(path.relative_to(project_root))
                        files[rel] = path.read_text(encoding="utf-8")
                    except (UnicodeDecodeError, PermissionError):
                        continue

        logger.info("files_read", agent=self.name, count=len(files))
        return files

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the agent's system prompt."""
        settings = get_settings()
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]
        return chat_completion(
            messages,
            temperature=0.2,
            max_tokens=settings.agent_max_tokens_per_request,
        )

    def _analyze_batch(self, files: dict[str, str]) -> list[Finding]:
        """Send files to LLM and parse findings."""
        prompt = self._build_analysis_prompt(files)
        raw = self._call_llm(prompt)
        return self._parse_findings(raw, files)

    def _generate_summary(
        self, findings: list[Finding]
    ) -> tuple[str, list[str]]:
        """Ask LLM to summarize findings and generate recommendations."""
        findings_text = "\n".join(
            f"- [{f.severity.value.upper()}] {f.file_path}: {f.description}"
            for f in findings
        )
        prompt = (
            f"Based on the following {len(findings)} findings from a {self.name} analysis, "
            f"provide:\n"
            f"1. A concise 2-3 sentence summary\n"
            f"2. Top 5 actionable recommendations\n\n"
            f"Findings:\n{findings_text}\n\n"
            f'Respond in JSON: {{"summary": "...", "recommendations": ["...", ...]}}'
        )
        raw = self._call_llm(prompt)
        try:
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(raw)
            return data.get("summary", ""), data.get("recommendations", [])
        except (json.JSONDecodeError, KeyError):
            return f"Analysis found {len(findings)} findings.", []

    def _save_report(self, report: AgentReport) -> Path:
        """Save report as JSON to the configured report directory."""
        settings = get_settings()
        report_dir = Path(settings.agent_report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = report.timestamp.replace(":", "-").replace("+", "_")
        filename = f"{report.agent_name}_{timestamp}.json"
        path = report_dir / filename

        path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
        logger.info("report_saved", agent=report.agent_name, path=str(path))
        return path

    @staticmethod
    def _parse_json_findings(raw: str) -> list[dict]:
        """Extract a JSON array of findings from LLM output."""
        arr_match = re.search(r"\[[\s\S]*\]", raw)
        if arr_match:
            try:
                return json.loads(arr_match.group())
            except json.JSONDecodeError:
                pass
        obj_match = re.search(r"\{[\s\S]*\}", raw)
        if obj_match:
            try:
                data = json.loads(obj_match.group())
                if isinstance(data.get("findings"), list):
                    return data["findings"]
            except json.JSONDecodeError:
                pass
        return []
