"""
State Ledger data model.

Defines the hierarchical Pydantic structure that is the single source of truth for
an incident. The orchestrator reads and reasons only over this object; only parsed
primitives that satisfy the type constraints below may populate these fields (enforced
by the State Mutation Layer). Literal fields lock each field to a finite state set, and
`loop_count` is hard-bounded so the schema itself participates in circuit-breaking.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict
from uuid import UUID


class IncidentMetadata(BaseModel):
    """Identity and lifecycle counters for a single incident."""

    id: UUID
    type: Literal["SUPPLIER_DELAY", "QUALITY_FAILURE", "BANKRUPTCY"] = "SUPPLIER_DELAY"
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "CRITICAL"
    # Circuit breaker forces escalation when this exceeds 10.
    loop_count: int = Field(default=0, ge=0, le=11)


class BusinessContext(BaseModel):
    """Enterprise entities the incident is anchored to (SKU, supplier, PO, contract)."""

    target_sku: str
    impacted_plants: List[str] = []
    primary_supplier_id: str
    active_contract_id: str
    current_purchase_order_id: str
    # Per-diem late-delivery penalty rate parsed from the contract (e.g. 0.03 = 3%).
    contracted_penalty_rate: float = 0.0


class ImpactMetrics(BaseModel):
    """Quantified operational and financial exposure. Populated by the Decision Helpers."""

    inventory_days_remaining: int
    production_shutdown_hours: int
    revenue_at_risk_usd: float

    # Mitigation FEASIBILITY signals — which options are possible this incident. These gate
    # strategy SELECTION deterministically, independent of the desirability scores: the
    # required `replacement_order_qty` is compared against these finite resources, so a
    # larger shortfall organically closes cheaper options.
    #   * transferable_units: surplus PLANT-1 can move to PLANT-2 (INTERNAL_TRANSFER).
    #   * air_freight_available: whether the lane can be expedited by air at all.
    #   * air_freight_capacity_units: finite air cargo capacity (AIR_FREIGHT).
    transferable_units: int = 350
    air_freight_available: bool = True
    air_freight_capacity_units: int = 420

    # Replacement quantity this incident must cover — the same value the spend guardrail
    # multiplies by unit price. Held on the ledger so the LLM can apply the feasibility rule.
    replacement_order_qty: int = 0

    # Observed shipment delay (days) written by query_shipment_tracking. simulate_finance is
    # the sole authority that converts it into downtime; kept separate from
    # production_shutdown_hours so an observation never clobbers the incident baseline.
    delay_days: int = 0

    # Freight-market cost multiplier (1.0 = baseline); scales expedited-freight scoring.
    market_freight_index_multiplier: float = 1.0
    # Financial primitives written by simulate_finance so the ledger captures the incident's
    # exposure, not just the static baseline revenue_at_risk_usd.
    daily_penalty_usd: float = 0.0
    projected_total_loss_usd: float = 0.0


class MitigationState(BaseModel):
    """Active resolution strategy and the status of downstream workflows."""

    active_strategy: Literal["NONE", "ALT_SUPPLIER", "INTERNAL_TRANSFER", "AIR_FREIGHT"] = "NONE"
    # Deterministic 1-100 scores keyed by strategy name, written by score_strategy().
    strategy_scores: Dict[str, float] = {}
    rfq_status: Literal["IDLE", "PENDING", "RECEIVED", "EXPIRED"] = "IDLE"
    negotiation_status: Literal["IDLE", "IN_PROGRESS", "SUCCESS", "FAILED"] = "IDLE"
    # Final negotiated primitives (raw vendor dialogue stays in incident_execution.log).
    agreed_unit_price_usd: float = 0.0
    agreed_lead_time_days: int = 0
    # Winning vendor from a competitive negotiation; None until a supplier is selected.
    agreed_supplier_id: Optional[str] = None


class SystemStatus(BaseModel):
    """Guardrail and goal state. Owned by guardrail code; the LLM cannot write these."""

    guardrail_status: Literal["PASSED", "BREACHED"] = "PASSED"
    goal_achieved: bool = False
    escalation_reason: Optional[str] = None


class StateLedger(BaseModel):
    """Root State Ledger the Incident Commander evaluates each loop to select its action."""

    metadata: IncidentMetadata
    context: BusinessContext
    metrics: ImpactMetrics
    mitigation: MitigationState
    status: SystemStatus


__all__ = [
    "IncidentMetadata",
    "BusinessContext",
    "ImpactMetrics",
    "MitigationState",
    "SystemStatus",
    "StateLedger",
]
