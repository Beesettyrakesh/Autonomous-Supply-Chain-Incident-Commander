"""
ledger_store.py
===============
In-memory storage driver for the State Ledger.

This module is the concrete implementation of the "single source of truth" storage
concept described in Section 3. It holds exactly one live `StateLedger` instance per
incident and mediates ALL mutations to it. Components in the graph must never mutate
the ledger by passing ad-hoc objects around — they go through this store, which is the
seat of the State Mutation Layer's contract.

Guarantees provided here:
- Only typed primitives that survive Pydantic validation can enter the ledger.
- Every mutation increments an internal revision counter for auditability.
- Raw / unstructured tool output is NEVER stored here; it is appended to the external
  `incident_execution.log` via `append_raw_log()` and is not natively readable by the
  Orchestrator (Section 3.2).
"""

from __future__ import annotations

import asyncio
import copy
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, cast
from uuid import UUID, uuid4



from schema import (
    StateLedger,
    IncidentMetadata,
    BusinessContext,
    ImpactMetrics,
    MitigationState,
    SystemStatus,
)

# Isolated raw-detail sink. The Orchestrator cannot read this natively (Section 3.2).
RAW_LOG_PATH = Path(__file__).parent / "incident_execution.log"

# Control characters that must NEVER reach the flat audit log verbatim. Adversarial input
# (e.g. an injection blob, a poisoned vendor reply) could otherwise embed carriage returns /
# line feeds (\r \n) to FORGE fake log lines, or ANSI/ESC sequences to hide/rewrite terminal
# output when the log is `cat`-ed. We neutralize ALL C0 control bytes (0x00-0x1f — this
# INCLUDES tab, CR, LF and ESC) plus DEL (0x7f) at the single write boundary, so every call
# site is protected (log-forging defense, §5.1). The record's only newline is the trailing
# "\n" the writer appends itself.
_LOG_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")




class LedgerStore:
    """
    Thread-safe, in-memory holder + mutator for a single incident's StateLedger.

    A callback hook (`on_mutation`) is exposed so downstream watchers — e.g. the
    Streamlit dashboard or the async negotiation callback — can react to state changes
    without polling.
    """

    def __init__(self) -> None:
        self._ledger: Optional[StateLedger] = None
        self._revision: int = 0
        self._lock = threading.RLock()
        # Optional observer invoked after every successful mutation.
        self.on_mutation: Optional[Callable[[StateLedger, int], None]] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def init_incident(
        self,
        *,
        target_sku: str,
        primary_supplier_id: str,
        active_contract_id: str,
        current_purchase_order_id: str,
        impacted_plants: Optional[list[str]] = None,
        inventory_days_remaining: int = 0,
        production_shutdown_hours: int = 0,
        revenue_at_risk_usd: float = 0.0,
        transferable_units: int = 350,
        air_freight_available: bool = True,
        air_freight_capacity_units: int = 420,
        replacement_order_qty: int = 0,
        delay_days: int = 9,
        incident_type: str = "SUPPLIER_DELAY",


        severity: str = "CRITICAL",
        incident_id: Optional[UUID] = None,
    ) -> StateLedger:
        """
        Corresponds to the INIT STATE LEDGER node: sets IDs, SKUs, and LoopCount=0.
        Returns the freshly created, validated ledger.

        `transferable_units`, `air_freight_available`, and `air_freight_capacity_units` seed
        the mitigation FEASIBILITY signals (which options are even possible this incident) —
        these gate strategy SELECTION deterministically, independent of the desirability
        scores. The agent's choice EMERGES from comparing the required `replacement_order_qty`
        against these finite resources (no scenario switch): a small order is covered by the
        internal transfer surplus; a larger one exceeds it and is expedited by air; a still
        larger one exceeds even the air capacity and must be sourced from an ALTERNATE supplier
        (negotiation + spend-authority guardrail + HITL). `delay_days` seeds the observed
        shipment slip that drives the dynamic financial-exposure computation in simulate_finance.
        """

        with self._lock:
            ledger = StateLedger(
                metadata=IncidentMetadata(
                    id=incident_id or uuid4(),
                    type=incident_type,  # type: ignore[arg-type]  # validated by Literal
                    severity=severity,  # type: ignore[arg-type]
                    loop_count=0,
                ),
                context=BusinessContext(
                    target_sku=target_sku,
                    impacted_plants=impacted_plants or [],
                    primary_supplier_id=primary_supplier_id,
                    active_contract_id=active_contract_id,
                    current_purchase_order_id=current_purchase_order_id,
                ),
                metrics=ImpactMetrics(
                    inventory_days_remaining=inventory_days_remaining,
                    production_shutdown_hours=production_shutdown_hours,
                    revenue_at_risk_usd=revenue_at_risk_usd,
                    transferable_units=transferable_units,
                    air_freight_available=air_freight_available,
                    air_freight_capacity_units=air_freight_capacity_units,
                    replacement_order_qty=replacement_order_qty,
                    delay_days=delay_days,
                ),


                mitigation=MitigationState(),
                status=SystemStatus(),
            )

            self._ledger = ledger
            self._revision = 0
            self._notify()
            return self.snapshot()

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def snapshot(self) -> StateLedger:
        """
        Return a deep copy of the current ledger. Callers get an immutable-by-convention
        view: mutating the returned object never affects stored state — the only way to
        change state is via `mutate()`.
        """
        with self._lock:
            if self._ledger is None:
                raise RuntimeError("No active incident. Call init_incident() first.")
            return self._ledger.model_copy(deep=True)

    def snapshot_dict(self) -> Dict[str, Any]:
        """JSON-safe dict snapshot for tools that expect a plain state dictionary."""
        return self.snapshot().model_dump(mode="json")

    @property
    def revision(self) -> int:
        return self._revision

    # ------------------------------------------------------------------ #
    # Writes (State Mutation Layer contract)
    # ------------------------------------------------------------------ #
    def mutate(self, patch: Dict[str, Any]) -> StateLedger:
        """
        Apply a partial, nested update to the ledger.

        `patch` is a nested dict keyed by top-level section, e.g.:
            {"metrics": {"revenue_at_risk_usd": 4200.0},
             "mitigation": {"active_strategy": "ALT_SUPPLIER"}}

        The merged result is re-validated through the full Pydantic model, so any value
        that violates a type or Literal constraint raises and the mutation is rejected —
        this is the deterministic enforcement point for the State Mutation Layer.
        """
        with self._lock:
            if self._ledger is None:
                raise RuntimeError("No active incident. Call init_incident() first.")

            merged = self._deep_merge(self._ledger.model_dump(), patch)
            # Re-validate the entire ledger; rejects bad primitives atomically.
            validated = StateLedger.model_validate(merged)
            self._ledger = validated
            self._revision += 1
            self._notify()
            return self.snapshot()

    def increment_loop(self) -> StateLedger:
        """Convenience mutation used by the UPDATE STATE LEDGER node (LoopCount++)."""
        current = self.snapshot()
        return self.mutate(
            {"metadata": {"loop_count": current.metadata.loop_count + 1}}
        )

    # ------------------------------------------------------------------ #
    # Raw log sink (Section 3.2) — never enters the ledger
    # ------------------------------------------------------------------ #
    @staticmethod
    def append_raw_log(source: str, raw_payload: str) -> None:
        """
        Append verbose / unstructured tool output to the isolated execution log.
        The Orchestrator cannot read this natively; only an explicit lookup tool may.

        SECURITY (log-forging defense, §5.1): both `source` and `raw_payload` may carry
        adversarial content (LLM output, vendor replies, blocked injection blobs). We strip
        embedded control characters — CR/LF (which would forge extra log lines) and ANSI/ESC
        sequences (which would corrupt terminal rendering) — BEFORE writing. This is the
        single write boundary, so EVERY call site is protected here (not per-f-string). The
        record still occupies exactly one line; the trailing "\\n" is the only newline.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        safe_source = _LOG_CONTROL_CHARS_RE.sub(" ", str(source))
        safe_payload = _LOG_CONTROL_CHARS_RE.sub(" ", str(raw_payload))
        with RAW_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"[{timestamp}] [{safe_source}] {safe_payload}\n")

    @staticmethod
    async def append_raw_log_async(source: str, raw_payload: str) -> None:
        """Non-blocking variant of `append_raw_log` for use inside async coroutines.

        The synchronous `append_raw_log` does a blocking file write. Called from within the
        concurrent negotiation sub-graphs (multiple vendors under `asyncio.gather`), a raw
        blocking write would briefly stall the event loop and serialize otherwise-concurrent
        LLM network calls. This wrapper offloads the write to a worker thread via
        `asyncio.to_thread`, keeping the loop free so simultaneous vendor calls stay truly
        concurrent. Same control-char sanitization applies (it reuses `append_raw_log`).
        """
        await asyncio.to_thread(LedgerStore.append_raw_log, source, raw_payload)


    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _notify(self) -> None:
        if self.on_mutation is not None and self._ledger is not None:
            # Hand watchers a deep copy so they cannot corrupt canonical state.
            self.on_mutation(self._ledger.model_copy(deep=True), self._revision)

    @staticmethod
    def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge `patch` into a copy of `base` (patch wins on leaves)."""
        result: Dict[str, Any] = copy.deepcopy(base)
        for key, value in patch.items():
            existing = result.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                # Explicit casts keep the recursive call fully typed for strict checkers.
                result[key] = LedgerStore._deep_merge(
                    cast(Dict[str, Any], existing), cast(Dict[str, Any], value)
                )
            else:
                result[key] = value
        return result


# Module-level singleton the rest of the system shares by default.
STORE = LedgerStore()


__all__ = ["LedgerStore", "STORE", "RAW_LOG_PATH"]
