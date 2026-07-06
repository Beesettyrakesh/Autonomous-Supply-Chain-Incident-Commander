"""Decision Helper tests — the frozen finance and strategy-score numbers."""

from __future__ import annotations

from decision_helpers import policy_check, score_strategy, simulate_finance

# Ledger snapshot shape the helpers read (matches the demo incident).
_SNAPSHOT = {
    "metadata": {"severity": "CRITICAL"},
    "context": {"contracted_penalty_rate": 0.03},
    "metrics": {
        "revenue_at_risk_usd": 75000.0,
        "inventory_days_remaining": 2,
        "production_shutdown_hours": 48,
        "market_freight_index_multiplier": 1.0,
    },
}


def test_simulate_finance_delay_9() -> None:
    r = simulate_finance(delay_days=9, state_ledger_snapshot=_SNAPSHOT)
    assert r["projected_total_loss_usd"] == 357750.0
    assert r["daily_penalty_usd"] == 2250.0


def test_simulate_finance_delay_5() -> None:
    assert simulate_finance(5, _SNAPSHOT)["projected_total_loss_usd"] == 198750.0


def test_simulate_finance_delay_12() -> None:
    assert simulate_finance(12, _SNAPSHOT)["projected_total_loss_usd"] == 477000.0


def test_simulate_finance_delay_0_no_downtime() -> None:
    """Explicit 0 (on-time) is respected: only baseline revenue, no penalty/downtime."""
    r = simulate_finance(0, _SNAPSHOT)
    assert r["projected_total_loss_usd"] == 75000.0


def test_simulate_finance_fallback_to_ledger() -> None:
    """delay_days=None falls back to the ledger's observed slip (the offline-planner path).

    The orchestrator's offline planner calls simulate_finance with empty args, so the
    None -> metrics.delay_days fallback is the production path. Assert it reads the snapshot's
    baseline delay (9) and reproduces the frozen $357,750 loss — a direct regression guard for
    a bug that previously computed the loss off a missing delay.
    """
    snapshot = {**_SNAPSHOT, "metrics": {**_SNAPSHOT["metrics"], "delay_days": 9}}
    r = simulate_finance(delay_days=None, state_ledger_snapshot=snapshot)
    assert r["delay_days"] == 9
    assert r["projected_total_loss_usd"] == 357750.0


def test_score_alt_supplier() -> None:
    assert score_strategy("ALT_SUPPLIER", _SNAPSHOT)["composite_score"] == 60.35


def test_score_internal_transfer() -> None:
    assert score_strategy("INTERNAL_TRANSFER", _SNAPSHOT)["composite_score"] == 80.0


def test_score_air_freight() -> None:
    assert score_strategy("AIR_FREIGHT", _SNAPSHOT)["composite_score"] == 65.28


def test_policy_check_within_over_and_unapproved() -> None:
    assert policy_check("SUP-C", 19360.0) is True       # within $20k
    assert policy_check("SUP-C", 22000.0) is False      # over $20k
    assert policy_check("SUP-Z", 100.0) is False        # unapproved vendor
