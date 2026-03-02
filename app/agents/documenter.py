"""Documenter Agent — generates comprehensive documentation for the codebase."""

from __future__ import annotations

import json
import re

from app.agents.base import BaseAgent
from app.agents.models import Finding, Severity, AgentReport
from app.observability.logger import get_logger

logger = get_logger(__name__)


class DocumenterAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "documenter"

    @property
    def description(self) -> str:
        return "Generates comprehensive markdown documentation for the project"

    @property
    def _system_prompt(self) -> str:
        return (
            "You are a technical documentation specialist. Generate clear, "
            "comprehensive documentation for a Python/TypeScript codebase.\n\n"
            "For each file provided, document:\n"
            "1. **Module purpose**: What this module does\n"
            "2. **Key classes/functions**: Purpose, parameters, return values\n"
            "3. **API endpoints** (if applicable): Routes, methods, schemas\n"
            "4. **Dependencies**: What this module depends on\n\n"
            "Respond ONLY with a JSON object:\n"
            "{\n"
            '  "modules": [\n'
            "    {\n"
            '      "file_path": "...",\n'
            '      "purpose": "...",\n'
            '      "classes": [{"name": "...", "description": "...", "methods": ["..."]}],\n'
            '      "functions": [{"name": "...", "description": "...", "params": "...", "returns": "..."}],\n'
            '      "endpoints": [{"method": "...", "path": "...", "description": "..."}],\n'
            '      "dependencies": ["..."]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Be accurate — only document what actually exists in the code."
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
        return "Document the following source files:\n\n" + "\n\n".join(parts)

    def _parse_findings(self, raw: str, files: dict[str, str]) -> list[Finding]:
        findings = []
        try:
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(raw)

            for module in data.get("modules", []):
                findings.append(
                    Finding(
                        category="documentation",
                        severity=Severity.INFO,
                        file_path=module.get("file_path", "unknown"),
                        description=json.dumps(module, indent=2),
                        suggestion=module.get("purpose", ""),
                    )
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("documenter_parse_error", error=str(e))
            for file_path in files:
                findings.append(
                    Finding(
                        category="documentation",
                        severity=Severity.INFO,
                        file_path=file_path,
                        description=raw[:500],
                        suggestion="",
                    )
                )
        return findings

    def analyze(self) -> AgentReport:
        """Override to also generate a combined markdown doc."""
        report = super().analyze()

        md_parts = ["# Rocket Mortgage FAQ Bot — Technical Documentation\n"]
        md_parts.append(f"*Generated: {report.timestamp}*\n")

        for finding in report.findings:
            if finding.category == "documentation":
                try:
                    module = json.loads(finding.description)
                    md_parts.append(
                        f"\n## `{module.get('file_path', finding.file_path)}`\n"
                    )
                    md_parts.append(f"\n{module.get('purpose', '')}\n")
                    for cls in module.get("classes", []):
                        md_parts.append(f"\n### Class: `{cls['name']}`\n")
                        md_parts.append(f"{cls.get('description', '')}\n")
                    for func in module.get("functions", []):
                        md_parts.append(
                            f"\n### `{func['name']}({func.get('params', '')})`\n"
                        )
                        md_parts.append(f"{func.get('description', '')}\n")
                        if func.get("returns"):
                            md_parts.append(f"**Returns:** {func['returns']}\n")
                    for ep in module.get("endpoints", []):
                        md_parts.append(
                            f"\n### `{ep.get('method', 'GET')} {ep.get('path', '')}`\n"
                        )
                        md_parts.append(f"{ep.get('description', '')}\n")
                except (json.JSONDecodeError, KeyError):
                    md_parts.append(f"\n## `{finding.file_path}`\n")
                    md_parts.append(f"{finding.description[:300]}\n")

        report.metadata["markdown_doc"] = "\n".join(md_parts)
        return report

    def _generate_summary(
        self, findings: list[Finding]
    ) -> tuple[str, list[str]]:
        file_count = len(set(f.file_path for f in findings))
        summary = (
            f"Generated documentation for {file_count} source files "
            f"across the backend and frontend."
        )
        recommendations = [
            "Add module-level docstrings to files missing them",
            "Add type hints to function parameters that lack them",
            "Document configuration environment variables",
            "Add inline comments for complex business logic",
            "Create an architecture diagram showing component relationships",
        ]
        return summary, recommendations
