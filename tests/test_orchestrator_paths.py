"""End-to-end orchestrator tests for the four quantity-ladder paths (offline planner)."""

from __future__ import annotations

import asyncio

from ledger_store import STORE
from orchestrator import IncidentCommander


def _run_incident(order_quantity: int, *, approve: bool = True, delay_days: int = 9):  # type: ignore[no-untyped-def]
    """Seed the incident and run the commander offline; return the final ledger snapshot."""
    STORE.init_incident(
        target_sku="SKU-99",
        primary_supplier_id="SUP-A",
        active_contract_id="CTR-4471",
        current_purchase_order_id="PO-88123",
        impacted_plants=["PLANT-2"],
        inventory_days_remaining=2,
        production_shutdown_hours=48,
        revenue_at_risk_usd=75000.0,
        transferable_units=350,
        air_freight_available=True,
        air_freight_capacity_units=420,
        replacement_order_qty=order_quantity,
        delay_days=delay_days,
    )
    commander = IncidentCommander(
        order_quantity=order_quantity,
        human_decision=lambda _r: approve,  # sync HITL provider
    )
    asyncio.run(commander.run(verbose=False))
    return STORE.snapshot()


def test_qty_300_internal_transfer() -> None:
    led = _run_incident(300)
    assert led.mitigation.active_strategy == "INTERNAL_TRANSFER"
    assert led.status.guardrail_status == "PASSED"
    assert led.metrics.projected_total_loss_usd == 357750.0


def test_qty_400_air_freight() -> None:
    led = _run_incident(400)
    assert led.mitigation.active_strategy == "AIR_FREIGHT"
    assert led.status.guardrail_status == "PASSED"


def test_qty_440_alt_supplier_within_authority() -> None:
    led = _run_incident(440)
    assert led.mitigation.active_strategy == "ALT_SUPPLIER"
    assert led.mitigation.agreed_supplier_id == "SUP-C"
    assert led.mitigation.agreed_unit_price_usd == 44.0
    assert led.status.guardrail_status == "PASSED"
    assert led.mitigation.agreed_unit_price_usd * 440 == 19360.0


def test_qty_500_hitl_approve() -> None:
    led = _run_incident(500, approve=True)
    assert led.mitigation.active_strategy == "ALT_SUPPLIER"
    assert led.status.guardrail_status == "PASSED"
    assert led.status.escalation_reason == "human_approved_over_limit_spend"


def test_qty_500_hitl_reject_escalates_and_nulls_terms() -> None:
    led = _run_incident(500, approve=False)
    assert led.status.guardrail_status == "BREACHED"
    assert led.mitigation.active_strategy == "NONE"
    # No phantom PO: negotiated terms are nulled on rejection.
    assert led.mitigation.agreed_supplier_id is None
    assert led.mitigation.agreed_unit_price_usd == 0.0


def test_feasibility_backstop_blocks_infeasible_commit() -> None:
    """INTERNAL_TRANSFER at qty 500 exceeds the 350 surplus -> dispatch returns an error."""
    STORE.init_incident(
        target_sku="SKU-99",
        primary_supplier_id="SUP-A",
        active_contract_id="CTR-4471",
        current_purchase_order_id="PO-88123",
        impacted_plants=["PLANT-2"],
        inventory_days_remaining=2,
        production_shutdown_hours=48,
        revenue_at_risk_usd=75000.0,
        transferable_units=350,
        air_freight_available=True,
        air_freight_capacity_units=420,
        replacement_order_qty=500,
        delay_days=9,
    )
    commander = IncidentCommander(order_quantity=500)
    # _dispatch_tool is async (observation tools flow through the MCP client transport); the
    # feasibility backstop still resolves synchronously for commit_strategy.
    out = asyncio.run(commander._dispatch_tool("commit_strategy", {"strategy_type": "INTERNAL_TRANSFER"}))
    assert "error" in out["result"]
    assert "INFEASIBLE" in out["result"]["error"]

