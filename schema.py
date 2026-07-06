"""
schema.py
=========
The single, non-negotiable source of truth for the *State Ledger* data model of the
Autonomous Supply Chain Incident Commander.

This module implements the exact hierarchical Pydantic structure defined in Section 3.1
of the master handoff specification. The State Ledger is the ONLY object the Orchestrator
is permitted to read and reason over. No raw unstructured text, markdown, or conversational
tool output may ever populate these fields — only parsed primitives that satisfy the type
constraints below (enforced downstream by the State Mutation Layer).

Design notes:
- Literal[...] fields lock the reasoning space to a finite, deterministic set of states.
- loop_count is hard-bounded (ge=0, le=11) so the schema itself participates in circuit-breaking.
- revenue_at_risk_usd is a float because the Financial Spend Guardrail compares it to 5000.00.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict
from uuid import UUID


class IncidentMetadata(BaseModel):
    """Top-level identity + lifecycle counters for a single incident."""

    # Unique incident identifier assigned at INIT STATE LEDGER.
    id: UUID
    # The class of disruption we are commanding a response to.
    type: Literal["SUPPLIER_DELAY", "QUALITY_FAILURE", "BANKRUPTCY"] = "SUPPLIER_DELAY"
    # Business severity — drives escalation urgency, not spend authorization.
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "CRITICAL"
    # Hard-bounded loop counter. The circuit breaker forces escalation when this exceeds 10.
    loop_count: int = Field(default=0, ge=0, le=11)


class BusinessContext(BaseModel):
    """Enterprise entities the incident is anchored to (SKUs, suppliers, POs, contracts)."""

    target_sku: str
    impacted_plants: List[str] = []
    primary_supplier_id: str
    active_contract_id: str
    current_purchase_order_id: str
    # Per-diem late-delivery penalty rate parsed from the active contract (e.g. 0.03 = 3%).
    # Populated by the extract_contract_rules MCP tool; consumed by simulate_finance.
    contracted_penalty_rate: float = 0.0


class ImpactMetrics(BaseModel):
    """Quantified operational + financial exposure. Populated ONLY by Decision Helpers."""

    inventory_days_remaining: int
    production_shutdown_hours: int
    # Guarded value: the Financial Spend Guardrail hard-forks to HUMAN TAKEOVER if this > 5000.00.
    revenue_at_risk_usd: float
    # --- Mitigation FEASIBILITY signals (single source of truth for which options are even
    # possible this incident). These gate strategy SELECTION deterministically — separate from
    # the desirability SCORES in score_strategy. The agent's autonomous choice EMERGES from
    # comparing the required `replacement_order_qty` against these finite resources — there is
    # NO scenario switch steering the outcome; a bigger shortfall organically closes options:
    #   * transferable_units: surplus PLANT-1 can move to PLANT-2. INTERNAL_TRANSFER is only
    #     feasible when this surplus covers the required order quantity.
    #   * air_freight_available: whether the delayed PO can be expedited by air on this lane
    #     at all (a hard on/off for the lane).
    #   * air_freight_capacity_units: the FINITE air cargo capacity available to expedite this
    #     PO. AIR_FREIGHT is only feasible when this capacity covers the required quantity —
    #     realistic, since aircraft have limited hold volume. So as the order quantity grows,
    #     the agent escalates INTERNAL_TRANSFER -> AIR_FREIGHT -> ALT_SUPPLIER purely on volume.
    transferable_units: int = 350
    air_freight_available: bool = True
    air_freight_capacity_units: int = 420

    # The replacement order quantity this incident must cover — the SAME value the spend
    # guardrail multiplies by unit price. Held on the ledger (not just in the orchestrator)
    # so the LLM can read it and apply the INTERNAL_TRANSFER feasibility rule
    # (transferable_units >= replacement_order_qty) instead of guessing.
    replacement_order_qty: int = 0


    # Observed shipment delay (days) written by query_shipment_tracking. This is the RAW
    # observation; simulate_finance is the SOLE authority that converts it into downtime
    # (only the delay days BEYOND inventory_days_remaining incur shutdown). Kept separate
    # from production_shutdown_hours so an observation never clobbers the incident baseline.
    delay_days: int = 0

    # Live freight-market cost multiplier (1.0 = baseline). Scales expedited-freight cost
    # scoring in score_strategy; sourced from observation tools / market telemetry.
    market_freight_index_multiplier: float = 1.0
    # Computed financial impact primitives, written by simulate_finance via the State
    # Mutation Layer so the ledger (single source of truth) captures the incident's
    # financial exposure — not just the static baseline revenue_at_risk_usd.
    daily_penalty_usd: float = 0.0
    projected_total_loss_usd: float = 0.0


class MitigationState(BaseModel):
    """Tracks which resolution strategy is active and the status of downstream workflows."""

    active_strategy: Literal["NONE", "ALT_SUPPLIER", "INTERNAL_TRANSFER", "AIR_FREIGHT"] = "NONE"
    # Deterministic 1-100 scores keyed by strategy name, written by score_strategy().
    strategy_scores: Dict[str, float] = {}
    rfq_status: Literal["IDLE", "PENDING", "RECEIVED", "EXPIRED"] = "IDLE"
    # Flipped to IN_PROGRESS when the async negotiation sub-graph takes over.
    negotiation_status: Literal["IDLE", "IN_PROGRESS", "SUCCESS", "FAILED"] = "IDLE"
    # Final negotiated primitives written by the async negotiation sub-graph (only the
    # parsed scalars enter the ledger; raw vendor dialogue stays in incident_execution.log).
    agreed_unit_price_usd: float = 0.0
    agreed_lead_time_days: int = 0
    # Winning vendor from a competitive (parallel) negotiation. Optional[str] — None until a
    # supplier is selected. Enables downstream ERP/PO integration to know WHICH vendor won.
    agreed_supplier_id: Optional[str] = None


class SystemStatus(BaseModel):
    """Guardrail + goal state. The LLM cannot write these directly; guardrail code owns them."""

    guardrail_status: Literal["PASSED", "BREACHED"] = "PASSED"
    goal_achieved: bool = False
    escalation_reason: Optional[str] = None


class StateLedger(BaseModel):
    """
    The root State Ledger. This composite object is the single source of truth the
    Incident Commander evaluates each loop to select its Next Best Action.
    """

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
