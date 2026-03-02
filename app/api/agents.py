"""API endpoints for the multi-agent analysis system."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app.agents.coordinator import AgentCoordinator
from app.config import get_settings
from app.core.llm import BudgetExceededError
from app.middleware.auth import verify_api_key
from app.middleware.rate_limit import limiter
from app.observability.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/agents", tags=["agents"])

_coordinator = AgentCoordinator()


def _run_agent(agent_name: str) -> dict:
    """Run a single agent, save report, return results."""
    report = _coordinator.run_single(agent_name)
    agent_instance = next(
        a for a in _coordinator.agents if a.name == agent_name
    )
    agent_instance._save_report(report)
    return report.to_dict()


@router.post("/review")
@limiter.limit(settings.agent_rate_limit)
async def run_code_review(
    request: Request,
    _api_key: str | None = Depends(verify_api_key),
):
    """Run the code review agent."""
    logger.info("agent_api_request", agent="code_reviewer")
    try:
        return _run_agent("code_reviewer")
    except BudgetExceededError:
        raise HTTPException(status_code=503, detail="Token budget exceeded")


@router.post("/document")
@limiter.limit(settings.agent_rate_limit)
async def run_documenter(
    request: Request,
    _api_key: str | None = Depends(verify_api_key),
):
    """Run the documenter agent."""
    logger.info("agent_api_request", agent="documenter")
    try:
        return _run_agent("documenter")
    except BudgetExceededError:
        raise HTTPException(status_code=503, detail="Token budget exceeded")


@router.post("/tech-debt")
@limiter.limit(settings.agent_rate_limit)
async def run_tech_debt(
    request: Request,
    _api_key: str | None = Depends(verify_api_key),
):
    """Run the tech debt analyzer agent."""
    logger.info("agent_api_request", agent="tech_debt")
    try:
        return _run_agent("tech_debt")
    except BudgetExceededError:
        raise HTTPException(status_code=503, detail="Token budget exceeded")


@router.post("/full")
@limiter.limit(settings.agent_rate_limit)
async def run_full_analysis(
    request: Request,
    _api_key: str | None = Depends(verify_api_key),
):
    """Run all agents via the coordinator."""
    logger.info("agent_api_request", agent="coordinator")
    try:
        report = _coordinator.run_all()
        _coordinator.save_report(report)
        return {
            "timestamp": report.timestamp,
            "executive_summary": report.executive_summary,
            "total_findings": report.total_findings,
            "findings_by_severity": report.findings_by_severity,
            "top_recommendations": report.top_recommendations,
            "agent_reports": report.agent_reports,
            "metadata": report.metadata,
        }
    except BudgetExceededError:
        raise HTTPException(status_code=503, detail="Token budget exceeded")


@router.get("/reports")
@limiter.limit(settings.rate_limit_default)
async def list_reports(
    request: Request,
    _api_key: str | None = Depends(verify_api_key),
):
    """List all saved agent reports."""
    report_dir = Path(settings.agent_report_dir)
    if not report_dir.exists():
        return {"reports": []}

    reports = []
    for path in sorted(report_dir.glob("*.json"), reverse=True):
        reports.append(
            {"filename": path.name, "size_bytes": path.stat().st_size}
        )
    return {"reports": reports}


@router.get("/reports/{filename}")
@limiter.limit(settings.rate_limit_default)
async def get_report(
    request: Request,
    filename: str,
    _api_key: str | None = Depends(verify_api_key),
):
    """Get a specific saved report by filename."""
    report_dir = Path(settings.agent_report_dir)
    path = report_dir / filename

    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Report not found")

    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    return JSONResponse(content=json.loads(path.read_text()))
