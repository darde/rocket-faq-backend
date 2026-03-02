"""Token usage tracking with configurable daily/monthly budgets."""

from __future__ import annotations

import threading
from datetime import datetime, date
from dataclasses import dataclass, field

from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)

# Approximate pricing per 1M tokens for common models via OpenRouter
MODEL_PRICING = {
    "google/gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
}

DEFAULT_PRICING = {"input": 0.50, "output": 1.50}


@dataclass
class UsageRecord:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    request_count: int = 0


@dataclass
class UsageTracker:
    """In-memory token usage tracker with daily budget enforcement."""

    _daily_usage: dict[str, UsageRecord] = field(default_factory=dict)
    _monthly_usage: dict[str, UsageRecord] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _today_key(self) -> str:
        return date.today().isoformat()

    def _month_key(self) -> str:
        return datetime.now().strftime("%Y-%m")

    def _estimate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
        cost = (
            prompt_tokens * pricing["input"]
            + completion_tokens * pricing["output"]
        ) / 1_000_000
        return round(cost, 6)

    def record_usage(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> dict:
        """Record token usage and return usage info including budget status."""
        cost = self._estimate_cost(model, prompt_tokens, completion_tokens)
        total = prompt_tokens + completion_tokens

        with self._lock:
            day_key = self._today_key()
            if day_key not in self._daily_usage:
                self._daily_usage[day_key] = UsageRecord()
            daily = self._daily_usage[day_key]
            daily.prompt_tokens += prompt_tokens
            daily.completion_tokens += completion_tokens
            daily.total_tokens += total
            daily.estimated_cost_usd += cost
            daily.request_count += 1

            month_key = self._month_key()
            if month_key not in self._monthly_usage:
                self._monthly_usage[month_key] = UsageRecord()
            monthly = self._monthly_usage[month_key]
            monthly.prompt_tokens += prompt_tokens
            monthly.completion_tokens += completion_tokens
            monthly.total_tokens += total
            monthly.estimated_cost_usd += cost
            monthly.request_count += 1

            # Clean old daily entries (keep only last 7 days)
            old_keys = [
                k
                for k in self._daily_usage
                if k < day_key and len(self._daily_usage) > 7
            ]
            for k in old_keys:
                del self._daily_usage[k]

        logger.info(
            "token_usage_recorded",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            daily_total_tokens=daily.total_tokens,
            daily_cost_usd=round(daily.estimated_cost_usd, 4),
        )

        return {
            "request_cost_usd": cost,
            "daily_total_tokens": daily.total_tokens,
            "daily_cost_usd": round(daily.estimated_cost_usd, 4),
            "monthly_total_tokens": monthly.total_tokens,
            "monthly_cost_usd": round(monthly.estimated_cost_usd, 4),
        }

    def check_budget(self) -> tuple[bool, str]:
        """Check if usage is within budget. Returns (within_budget, message)."""
        settings = get_settings()

        with self._lock:
            day_key = self._today_key()
            daily = self._daily_usage.get(day_key, UsageRecord())
            month_key = self._month_key()
            monthly = self._monthly_usage.get(month_key, UsageRecord())

        if (
            settings.daily_token_budget > 0
            and daily.total_tokens >= settings.daily_token_budget
        ):
            return False, (
                f"Daily token budget exceeded "
                f"({daily.total_tokens}/{settings.daily_token_budget})"
            )

        if (
            settings.monthly_token_budget > 0
            and monthly.total_tokens >= settings.monthly_token_budget
        ):
            return False, (
                f"Monthly token budget exceeded "
                f"({monthly.total_tokens}/{settings.monthly_token_budget})"
            )

        return True, "Within budget"

    def get_usage_summary(self) -> dict:
        """Return current usage summary for the stats endpoint."""
        settings = get_settings()

        with self._lock:
            day_key = self._today_key()
            daily = self._daily_usage.get(day_key, UsageRecord())
            month_key = self._month_key()
            monthly = self._monthly_usage.get(month_key, UsageRecord())

        return {
            "daily": {
                "date": day_key,
                "total_tokens": daily.total_tokens,
                "prompt_tokens": daily.prompt_tokens,
                "completion_tokens": daily.completion_tokens,
                "estimated_cost_usd": round(daily.estimated_cost_usd, 4),
                "request_count": daily.request_count,
                "budget_limit": settings.daily_token_budget,
            },
            "monthly": {
                "month": month_key,
                "total_tokens": monthly.total_tokens,
                "prompt_tokens": monthly.prompt_tokens,
                "completion_tokens": monthly.completion_tokens,
                "estimated_cost_usd": round(monthly.estimated_cost_usd, 4),
                "request_count": monthly.request_count,
                "budget_limit": settings.monthly_token_budget,
            },
        }


usage_tracker = UsageTracker()
