"""Code Review Agent — analyzes source files for quality, security, and best practices."""

from __future__ import annotations

from app.agents.base import BaseAgent
from app.agents.models import Finding, Severity


class CodeReviewAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "code_reviewer"

    @property
    def description(self) -> str:
        return "Analyzes source code for quality issues, security concerns, and best practice violations"

    @property
    def _system_prompt(self) -> str:
        return (
            "You are an expert code reviewer performing a thorough analysis of a "
            "Python/TypeScript codebase.\n\n"
            "Analyze the provided source files and identify:\n"
            "1. **Code Quality**: Poor naming, high complexity, missing error handling, code smells\n"
            "2. **Security**: Hardcoded secrets, missing input validation, injection risks\n"
            "3. **Best Practices**: Unused imports, missing type hints, inconsistent patterns\n"
            "4. **Improvements**: Concrete suggestions for better code\n\n"
            "For each finding, provide:\n"
            '- category: one of "quality", "security", "best_practices", "improvement"\n'
            '- severity: one of "info", "low", "medium", "high", "critical"\n'
            "- file_path: the file where the issue was found\n"
            "- description: clear description of the issue\n"
            "- suggestion: specific fix or improvement\n\n"
            "Respond ONLY with a JSON array of findings:\n"
            "[\n"
            '  {"category": "...", "severity": "...", "file_path": "...", '
            '"description": "...", "suggestion": "..."}\n'
            "]\n\n"
            "Be specific and actionable. Only report real problems."
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
            parts.append(f"### File: {path}\n```\n{content}\n```")
        return "Review the following source files:\n\n" + "\n\n".join(parts)

    def _parse_findings(self, raw: str, files: dict[str, str]) -> list[Finding]:
        raw_findings = self._parse_json_findings(raw)
        findings = []
        severity_map = {s.value: s for s in Severity}
        for item in raw_findings:
            sev = severity_map.get(item.get("severity", "info"), Severity.INFO)
            findings.append(
                Finding(
                    category=item.get("category", "quality"),
                    severity=sev,
                    file_path=item.get("file_path", "unknown"),
                    description=item.get("description", ""),
                    suggestion=item.get("suggestion", ""),
                    line_range=item.get("line_range", ""),
                )
            )
        return findings
