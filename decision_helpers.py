"""
decision_helpers.py
====================
Purely computational Python tools (Section 4.2) — the *Decision Helpers*.

These are deterministic mathematical engines. The LLM core is PHYSICALLY FORBIDDEN from
calculating financial impact or strategy scores qualitatively inside its prompt context;
it must delegate all such computation to these functions so results are reproducible and
auditable.

Day 3 scope: full deterministic computational engines. The public signatures and TypedDict
shapes are frozen from Day 1 and MUST NOT change; only the internal math is upgraded from
placeholder profiles to context-driven calculations that read real signals off the ledger.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict



class FinanceSimulationResult(TypedDict):
    """Explicit shape returned by `simulate_finance` — no `Any` leakage."""

    delay_days: int
    revenue_at_risk_usd: float
    daily_penalty_usd: float
    projected_total_loss_usd: float


class StrategyScoreResult(TypedDict):
    """Explicit shape returned by `score_strategy` — deterministic 1-100 ratings."""

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
    Calculate concrete cash-flow impact and downtime penalty parameters.

    Args:
        delay_days: Number of days the shipment / resolution is delayed (>= 0). Pass None
            to fall back to the ledger's observed `metrics.delay_days`. IMPORTANT: an
            explicit 0 (on-time shipment) is RESPECTED — only None triggers the fallback.

        state_ledger_snapshot: A JSON-safe dict snapshot of the current StateLedger. The
            baseline revenue exposure (`metrics.revenue_at_risk_usd`) and the per-diem
            penalty rate (`context.contracted_penalty_rate`) are read directly from it —
            no coefficients are hardcoded here.

    Returns:
        A strictly-typed dict of computed financial primitives, e.g.:
            {
                "delay_days": int,
                "revenue_at_risk_usd": float,
                "daily_penalty_usd": float,
                "projected_total_loss_usd": float,
            }

    Economic model (all inputs read from the ledger; no hardcoded coefficients):
    - Contract penalty accrual: `revenue_at_risk * contracted_penalty_rate` per delay day.
    - Downtime cost: production only halts once on-hand inventory is exhausted, so ONLY the
      delay days BEYOND `inventory_days_remaining` incur shutdown cost. That shutdown cost
      per day is derived from the ledger's `production_shutdown_hours` (hours of lost output
      the incident represents) valued at the same revenue exposure basis.
    """
    # Default both params so an LLM turn that omits them can't raise TypeError and crash
    # the ReAct loop; an absent snapshot degrades safely to zeros rather than exploding.
    snapshot = state_ledger_snapshot or {}
    metrics = snapshot.get("metrics", {})
    context = snapshot.get("context", {})
    revenue_at_risk = float(metrics.get("revenue_at_risk_usd", 0.0))

    # Contract-derived rate replaces any hardcoded coefficient.
    contracted_penalty_rate = float(context.get("contracted_penalty_rate", 0.0))
    inventory_days_remaining = int(metrics.get("inventory_days_remaining", 0))
    production_shutdown_hours = int(metrics.get("production_shutdown_hours", 0))

    # Prefer an explicit delay_days argument; otherwise fall back to the delay observed by
    # query_shipment_tracking and recorded on the ledger (`metrics.delay_days`). This keeps
    # simulate_finance the SOLE authority on downtime math while still honoring the observed
    # delay when the caller didn't pass one. NOTE: we check `is None` (NOT falsiness) so an
    # explicit 0 (on-time shipment) is respected and NOT overwritten by the stale ledger value.
    if delay_days is None:
        delay_days = int(metrics.get("delay_days", 0))
    else:
        delay_days = int(delay_days)



    # 1) Contractual late-delivery penalty accrues for every delay day.
    daily_penalty_usd = round(revenue_at_risk * contracted_penalty_rate, 2)
    penalty_component = round(daily_penalty_usd * delay_days, 2)

    # 2) Downtime only starts after the on-hand inventory buffer is depleted.
    shutdown_days = max(0, delay_days - inventory_days_remaining)
    # Value each shutdown day using the incident's lost-output hours against revenue basis.
    # revenue_at_risk represents the exposure over the full production_shutdown_hours window,
    # so the hourly rate is revenue_at_risk / production_shutdown_hours (guard div-by-zero).
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
    Compute structural scoring values for a candidate mitigation strategy.

    Args:
        strategy_type: One of "ALT_SUPPLIER" | "INTERNAL_TRANSFER" | "AIR_FREIGHT".
        state_ledger_snapshot: A JSON-safe dict snapshot of the current StateLedger.

    Returns:
        Deterministic ratings on a 1-100 scale plus a blended composite, e.g.:
            {
                "strategy_type": str,
                "cost_score": float,      # higher = cheaper / better
                "time_score": float,      # higher = faster / better
                "risk_score": float,      # higher = safer / better
                "composite_score": float, # weighted blend used by the Orchestrator
            }

    Scoring is context-driven from real ledger signals (all deterministic):
    - cost_score: strategy base cost eroded by the live `market_freight_index_multiplier`
      (freight-heavy options suffer most when the freight index is high).
    - time_score: strategy base speed adjusted by URGENCY — when `inventory_days_remaining`
      is low, fast strategies are rewarded and slow ones penalized.
    - risk_score: strategy base reliability adjusted DOWN by incident `severity`
      (CRITICAL incidents make every option relatively riskier).
    """
    # Base per-strategy profiles: cost (higher=cheaper), speed, reliability on a 0-100 scale.
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

    # --- cost_score: base cost eroded by freight index (deterministic, clamped 0-100) ---
    cost_score = round(min(100.0, max(0.0, profile["cost"] / freight_multiplier)), 2)

    # --- time_score: base speed scaled by urgency ---------------------------------------
    # Urgency multiplier grows as the inventory buffer shrinks (2-day buffer -> ~1.15x).
    # Fast strategies gain, slow ones are relatively penalized once clamped.
    urgency_factor = 1.0 + max(0, (3 - inventory_days_remaining)) * 0.05
    time_score = round(min(100.0, max(0.0, profile["speed"] * urgency_factor)), 2)

    # --- risk_score: base reliability discounted by incident severity -------------------
    severity_discount = {
        "LOW": 1.00, "MEDIUM": 0.95, "HIGH": 0.90, "CRITICAL": 0.85,
    }.get(severity, 0.95)
    risk_score = round(min(100.0, max(0.0, profile["reliability"] * severity_discount)), 2)

    # Composite weighting: time-critical incidents favor speed, but keep it deterministic.
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
    Verify vendor authorization against corporate purchasing policy compliance.

    Args:
        supplier_id: The vendor identifier being evaluated for a spend.
        spend_amount: Proposed procurement spend in USD.
        per_transaction_cap_usd: The buyer's delegated spend-authority limit. Spend at or
            below this may be auto-approved by the agent; above it requires human sign-off.
            Defaults to the $20,000 delegated-authority limit; callers (e.g. the Financial
            Spend Guardrail) may override it so the threshold stays configurable.

    Returns:
        True if the supplier is authorized AND the spend is within the delegated authority
        limit; else False (which the guardrail translates into a HUMAN TAKEOVER pause).

    NOTE: This models a procurement manager's fixed approval authority. The realistic
    variable per incident is the ORDER QUANTITY (and thus the total spend), not this limit.
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
