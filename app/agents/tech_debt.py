"""Tech Debt Analyzer Agent — static analysis + LLM-powered insights."""

from __future__ import annotations

import re
from pathlib import Path

from app.agents.base import BaseAgent
from app.agents.models import Finding, Severity, AgentReport
from app.observability.logger import get_logger

logger = get_logger(__name__)

_TODO_PATTERN = re.compile(
    r"#\s*(TODO|FIXME|HACK|XXX|TEMP|WORKAROUND)\b", re.IGNORECASE
)
_HARDCODED_PATTERN = re.compile(
    r"""(?:['"])(?:https?://|/api/|localhost|127\.0\.0\.1|0\.0\.0\.0)""",
    re.IGNORECASE,
)

FILE_LINE_THRESHOLD = 150


class TechDebtAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "tech_debt"

    @property
    def description(self) -> str:
        return "Identifies technical debt, TODOs, complexity, missing tests, and hardcoded values"

    @property
    def _system_prompt(self) -> str:
        return (
            "You are a senior software architect analyzing a codebase for technical debt.\n\n"
            "Identify:\n"
            "1. **Complexity**: Deeply nested logic, long functions, god classes\n"
            "2. **Hardcoded values**: Strings, URLs, numbers that should be configurable\n"
            "3. **Code duplication**: Similar patterns that could be abstracted\n"
            "4. **Missing abstraction**: Repeated boilerplate\n"
            "5. **Dependency concerns**: Tight coupling, circular dependencies\n"
            "6. **Testability**: Code that is hard to test\n\n"
            "For each finding, provide:\n"
            '- category: one of "complexity", "hardcoded", "duplication", "abstraction", "dependency", "testability"\n'
            '- severity: one of "info", "low", "medium", "high"\n'
            "- file_path: the file where the issue exists\n"
            "- description: what the technical debt is\n"
            "- suggestion: concrete refactoring suggestion\n\n"
            "Respond ONLY with a JSON array:\n"
            "[\n"
            '  {"category": "...", "severity": "...", "file_path": "...", '
            '"description": "...", "suggestion": "..."}\n'
            "]\n\n"
            "Focus on real, impactful issues."
        )

    def _get_file_patterns(self) -> list[str]:
        return [
            "rocket-faq-backend/app/**/*.py",
            "rocket-faq-backend/scripts/**/*.py",
            "rocket-faq-frontend/src/**/*.ts",
            "rocket-faq-frontend/src/**/*.tsx",
        ]

    def _build_analysis_prompt(self, files: dict[str, str]) -> str:
        parts = []
        for path, content in files.items():
            line_count = content.count("\n") + 1
            parts.append(
                f"### File: {path} ({line_count} lines)\n```\n{content}\n```"
            )
        return (
            "Analyze these files for technical debt and improvement opportunities:\n\n"
            + "\n\n".join(parts)
        )

    def _parse_findings(self, raw: str, files: dict[str, str]) -> list[Finding]:
        raw_findings = self._parse_json_findings(raw)
        findings = []
        severity_map = {s.value: s for s in Severity}
        for item in raw_findings:
            sev = severity_map.get(item.get("severity", "low"), Severity.LOW)
            findings.append(
                Finding(
                    category=item.get("category", "complexity"),
                    severity=sev,
                    file_path=item.get("file_path", "unknown"),
                    description=item.get("description", ""),
                    suggestion=item.get("suggestion", ""),
                )
            )
        return findings

    def analyze(self) -> AgentReport:
        """Override to prepend static analysis before LLM analysis."""
        static_findings = self._static_analysis()
        logger.info("tech_debt_static_complete", findings=len(static_findings))

        report = super().analyze()

        report.findings = static_findings + report.findings
        report.metadata["static_findings_count"] = len(static_findings)
        report.metadata["llm_findings_count"] = (
            len(report.findings) - len(static_findings)
        )
        return report

    def _static_analysis(self) -> list[Finding]:
        """Regex-based static analysis — no LLM cost."""
        findings: list[Finding] = []
        all_files = self._read_project_files(self._get_file_patterns())

        for rel_path, content in all_files.items():
            lines = content.split("\n")
            line_count = len(lines)

            if line_count > FILE_LINE_THRESHOLD:
                findings.append(
                    Finding(
                        category="complexity",
                        severity=Severity.LOW,
                        file_path=rel_path,
                        description=(
                            f"Large file with {line_count} lines "
                            f"(threshold: {FILE_LINE_THRESHOLD})"
                        ),
                        suggestion="Consider splitting into smaller, focused modules.",
                    )
                )

            for i, line in enumerate(lines, 1):
                match = _TODO_PATTERN.search(line)
                if match:
                    tag = match.group(1).upper()
                    findings.append(
                        Finding(
                            category="todo",
                            severity=(
                                Severity.MEDIUM
                                if tag in ("FIXME", "HACK")
                                else Severity.LOW
                            ),
                            file_path=rel_path,
                            description=f"{tag} comment: {line.strip()[:100]}",
                            suggestion=f"Address or track this {tag} item.",
                            line_range=str(i),
                        )
                    )

                if "config" not in rel_path.lower() and ".env" not in rel_path:
                    hc_match = _HARDCODED_PATTERN.search(line)
                    if hc_match:
                        findings.append(
                            Finding(
                                category="hardcoded",
                                severity=Severity.LOW,
                                file_path=rel_path,
                                description=f"Hardcoded value: {line.strip()[:80]}",
                                suggestion="Move to configuration/environment variable.",
                                line_range=str(i),
                            )
                        )

        # Check for missing tests
        backend_root = Path(__file__).resolve().parent.parent.parent
        if not (backend_root / "tests").exists():
            findings.append(
                Finding(
                    category="missing_tests",
                    severity=Severity.HIGH,
                    file_path="rocket-faq-backend/tests/",
                    description="No test directory found. The project has no automated tests.",
                    suggestion="Create a tests/ directory with pytest tests for core modules.",
                )
            )

        frontend_root = backend_root.parent / "rocket-faq-frontend"
        if frontend_root.exists() and not (frontend_root / "src" / "__tests__").exists():
            findings.append(
                Finding(
                    category="missing_tests",
                    severity=Severity.MEDIUM,
                    file_path="rocket-faq-frontend/src/__tests__/",
                    description="No frontend test directory found.",
                    suggestion="Add Vitest tests for components and API client.",
                )
            )

        return findings
