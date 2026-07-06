"""State Ledger data-model tests: bounds and Literal enforcement."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from schema import ImpactMetrics, IncidentMetadata, MitigationState
from ledger_store import STORE


def test_loop_count_accepts_10_rejects_12() -> None:
    """loop_count is bounded le=11; the circuit breaker escalates past 10."""
    IncidentMetadata(id=uuid.uuid4(), loop_count=10)  # ok
    with pytest.raises(ValidationError):
        IncidentMetadata(id=uuid.uuid4(), loop_count=12)


def test_loop_count_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        IncidentMetadata(id=uuid.uuid4(), loop_count=-1)


def test_invalid_active_strategy_rejected() -> None:
    with pytest.raises(ValidationError):
        MitigationState(active_strategy="EXPEDITE")  # type: ignore[arg-type]


def test_invalid_severity_rejected() -> None:
    with pytest.raises(ValidationError):
        IncidentMetadata(id=uuid.uuid4(), severity="EXTREME")  # type: ignore[arg-type]


def test_default_metrics_seed() -> None:
    """A default ImpactMetrics carries the demo feasibility resources."""
    m = ImpactMetrics(
        inventory_days_remaining=2, production_shutdown_hours=48, revenue_at_risk_usd=75000.0
    )
    assert m.transferable_units == 350
    assert m.air_freight_capacity_units == 420
    assert m.air_freight_available is True


def test_init_incident_builds_valid_ledger(init_incident) -> None:  # type: ignore[no-untyped-def]
    ledger = init_incident(order_quantity=500, delay_days=9)
    assert ledger.context.target_sku == "SKU-99"
    assert ledger.metrics.replacement_order_qty == 500
    assert ledger.metrics.delay_days == 9
    assert ledger.metadata.loop_count == 0
    # STORE holds the same incident.
    assert STORE.snapshot().context.target_sku == "SKU-99"
