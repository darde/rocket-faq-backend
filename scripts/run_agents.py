#!/usr/bin/env python3
"""CLI entry point for the multi-agent analysis system."""

import argparse
import json
import sys
from pathlib import Path

# Allow imports from the app package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.observability.logger import setup_logging, get_logger
from app.agents.coordinator import AgentCoordinator

setup_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run AI agents to analyze the Rocket FAQ codebase"
    )
    parser.add_argument("--all", action="store_true", help="Run all agents")
    parser.add_argument(
        "--review", action="store_true", help="Run code review agent"
    )
    parser.add_argument(
        "--document", action="store_true", help="Run documenter agent"
    )
    parser.add_argument(
        "--tech-debt", action="store_true", help="Run tech debt analyzer"
    )
    args = parser.parse_args()

    if not any([args.all, args.review, args.document, args.tech_debt]):
        parser.print_help()
        sys.exit(1)

    coordinator = AgentCoordinator()

    if args.all:
        print("Running all agents...")
        report = coordinator.run_all()
        path = coordinator.save_report(report)
        print(f"\nExecutive Summary:\n{report.executive_summary}")
        print(f"\nTotal findings: {report.total_findings}")
        print(f"By severity: {report.findings_by_severity}")
        print(f"\nTop recommendations:")
        for i, rec in enumerate(report.top_recommendations, 1):
            print(f"  {i}. {rec}")
        print(f"\nFull report saved to: {path}")
    else:
        agents_to_run = []
        if args.review:
            agents_to_run.append("code_reviewer")
        if args.document:
            agents_to_run.append("documenter")
        if args.tech_debt:
            agents_to_run.append("tech_debt")

        for agent_name in agents_to_run:
            print(f"\nRunning {agent_name}...")
            report = coordinator.run_single(agent_name)

            agent_instance = next(
                a for a in coordinator.agents if a.name == agent_name
            )
            path = agent_instance._save_report(report)

            print(f"  Summary: {report.summary}")
            print(f"  Findings: {len(report.findings)}")
            if report.recommendations:
                print(f"  Top recommendations:")
                for i, rec in enumerate(report.recommendations[:3], 1):
                    print(f"    {i}. {rec}")
            print(f"  Report saved to: {path}")


if __name__ == "__main__":
    main()
