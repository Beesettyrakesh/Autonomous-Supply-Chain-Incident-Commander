"""
Decision Helpers — deterministic computational tools.

The LLM core must not compute financial impact or strategy scores qualitatively in its
prompt; it delegates all such computation to these pure functions so results are
reproducible and auditable. Signatures and TypedDict shapes are stable.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict


class FinanceSimulationResult(TypedDict):
    """Shape returned by `simulate_finance`."""

    delay_days: int
    revenue_at_risk_usd: float
    daily_penalty_usd: float
    projected_total_loss_usd: float


class StrategyScoreResult(TypedDict):
    """Shape returned by `score_strategy` — deterministic 1-100 ratings."""

    strategy_type: str
    cost_score: float
    time_score: float
    risk_score: float
    composite_score: float


def simulate_finance(
    delay_days: Optional[int] = None,
    state_ledger_snapshot: Optional[Dict[str, Any]] = None,
) -> FinanceSimulationResult:

    """
    Calculate the cash-flow impact and downtime penalty of a delay.

    Args:
        delay_days: Days the shipment/resolution is delayed (>= 0). Pass None to use the
            ledger's observed `metrics.delay_days`. An explicit 0 (on-time) is respected —
            only None triggers the fallback.
        state_ledger_snapshot: JSON-safe StateLedger snapshot. Baseline revenue exposure
            (`metrics.revenue_at_risk_usd`) and the penalty rate
            (`context.contracted_penalty_rate`) are read from it; no coefficients are hardcoded.

    Returns:
        A FinanceSimulationResult of computed financial primitives.

    Economic model:
    - Contract penalty accrual: revenue_at_risk * contracted_penalty_rate per delay day.
    - Downtime cost: production only halts once on-hand inventory is exhausted, so ONLY the
      delay days beyond `inventory_days_remaining` incur shutdown, valued against the same
      revenue exposure basis via `production_shutdown_hours`.
    """
    # Default both params so an LLM turn that omits them degrades safely to zeros rather
    # than raising and crashing the ReAct loop.
    snapshot = state_ledger_snapshot or {}
    metrics = snapshot.get("metrics", {})
    context = snapshot.get("context", {})
    revenue_at_risk = float(metrics.get("revenue_at_risk_usd", 0.0))

    contracted_penalty_rate = float(context.get("contracted_penalty_rate", 0.0))
    inventory_days_remaining = int(metrics.get("inventory_days_remaining", 0))
    production_shutdown_hours = int(metrics.get("production_shutdown_hours", 0))

    # Prefer an explicit delay_days; otherwise use the observed slip on the ledger. Check
    # `is None` (not falsiness) so an explicit 0 is respected and not overwritten.
    if delay_days is None:
        delay_days = int(metrics.get("delay_days", 0))
    else:
        delay_days = int(delay_days)

    # 1) Contractual late-delivery penalty accrues per delay day.
    daily_penalty_usd = round(revenue_at_risk * contracted_penalty_rate, 2)
    penalty_component = round(daily_penalty_usd * delay_days, 2)

    # 2) Downtime only starts after the on-hand inventory buffer is depleted.
    shutdown_days = max(0, delay_days - inventory_days_remaining)
    hourly_downtime_cost = (
        revenue_at_risk / production_shutdown_hours if production_shutdown_hours > 0 else 0.0
    )
    downtime_component = round(hourly_downtime_cost * 24 * shutdown_days, 2)

    # Total projected loss = baseline exposure + penalty accrual + post-buffer downtime.
    projected_total_loss_usd = round(
        revenue_at_risk + penalty_component + downtime_component, 2
    )
    return {
        "delay_days": delay_days,
        "revenue_at_risk_usd": revenue_at_risk,
        "daily_penalty_usd": daily_penalty_usd,
        "projected_total_loss_usd": projected_total_loss_usd,
    }


def score_strategy(strategy_type: str, state_ledger_snapshot: Dict[str, Any]) -> StrategyScoreResult:
    """
    Compute deterministic scoring values for a candidate mitigation strategy.

    Args:
        strategy_type: One of "ALT_SUPPLIER" | "INTERNAL_TRANSFER" | "AIR_FREIGHT".
        state_ledger_snapshot: JSON-safe StateLedger snapshot.

    Returns:
        A StrategyScoreResult with cost/time/risk scores (higher = better) and a weighted
        composite used by the orchestrator.

    Scoring is context-driven from ledger signals:
    - cost_score: base cost eroded by the live `market_freight_index_multiplier`.
    - time_score: base speed scaled up by urgency when `inventory_days_remaining` is low.
    - risk_score: base reliability discounted by incident `severity`.
    """
    # Base per-strategy profiles: cost (higher=cheaper), speed, reliability (0-100).
    baseline_profiles: Dict[str, Dict[str, float]] = {
        "ALT_SUPPLIER": {"cost": 70.0, "speed": 55.0, "reliability": 60.0},
        "INTERNAL_TRANSFER": {"cost": 90.0, "speed": 75.0, "reliability": 80.0},
        "AIR_FREIGHT": {"cost": 30.0, "speed": 95.0, "reliability": 70.0},
    }
    profile = baseline_profiles.get(
        strategy_type, {"cost": 0.0, "speed": 0.0, "reliability": 0.0}
    )

    metrics = state_ledger_snapshot.get("metrics", {})
    metadata = state_ledger_snapshot.get("metadata", {})
    freight_multiplier = float(metrics.get("market_freight_index_multiplier", 1.0)) or 1.0
    inventory_days_remaining = int(metrics.get("inventory_days_remaining", 0))
    severity = str(metadata.get("severity", "MEDIUM"))

    cost_score = round(min(100.0, max(0.0, profile["cost"] / freight_multiplier)), 2)

    # Urgency multiplier grows as the inventory buffer shrinks (2-day buffer -> ~1.05x).
    urgency_factor = 1.0 + max(0, (3 - inventory_days_remaining)) * 0.05
    time_score = round(min(100.0, max(0.0, profile["speed"] * urgency_factor)), 2)

    severity_discount = {
        "LOW": 1.00, "MEDIUM": 0.95, "HIGH": 0.90, "CRITICAL": 0.85,
    }.get(severity, 0.95)
    risk_score = round(min(100.0, max(0.0, profile["reliability"] * severity_discount)), 2)

    composite_score = round(
        0.35 * cost_score + 0.40 * time_score + 0.25 * risk_score,
        2,
    )
    return {
        "strategy_type": strategy_type,
        "cost_score": cost_score,
        "time_score": time_score,
        "risk_score": risk_score,
        "composite_score": composite_score,
    }


def policy_check(
    supplier_id: str,
    spend_amount: float,
    per_transaction_cap_usd: float = 20000.00,
) -> bool:
    """
    Verify vendor authorization and spend against purchasing policy.

    Args:
        supplier_id: The vendor being evaluated.
        spend_amount: Proposed procurement spend in USD.
        per_transaction_cap_usd: The buyer's delegated spend-authority limit; spend at or
            below it may be auto-approved, above it requires human sign-off. Overridable so
            the threshold stays configurable.

    Returns:
        True if the supplier is approved AND the spend is within the delegated authority;
        else False (which the guardrail translates into a HUMAN TAKEOVER pause).
    """
    approved_vendors = {"SUP-A", "SUP-B", "SUP-C"}
    return supplier_id in approved_vendors and 0.0 < spend_amount <= per_transaction_cap_usd


__all__ = [
    "simulate_finance",
    "score_strategy",
    "policy_check",
    "FinanceSimulationResult",
    "StrategyScoreResult",
]
