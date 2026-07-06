"""
In-memory storage driver for the State Ledger.

Holds exactly one live `StateLedger` per incident and mediates all mutations to it, so the
ledger is the single source of truth. Guarantees:
- Only typed primitives that pass Pydantic validation can enter the ledger.
- Every mutation increments a revision counter for auditability.
- Raw/unstructured tool output never enters the ledger; it goes to the isolated
  `incident_execution.log` via `append_raw_log()`.
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

# Isolated raw-detail sink; not natively readable by the orchestrator.
RAW_LOG_PATH = Path(__file__).parent / "incident_execution.log"

# Control characters neutralized before writing to the audit log. Stripping C0 bytes
# (0x00-0x1f, incl. CR/LF/ESC) plus DEL (0x7f) prevents log forging (CR/LF injecting fake
# lines) and terminal corruption (ANSI/ESC sequences) from adversarial input. Applied at the
# single write boundary so every call site is protected.
_LOG_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


class LedgerStore:
    """
    Thread-safe, in-memory holder + mutator for a single incident's StateLedger.

    An `on_mutation` callback lets downstream watchers (dashboard, negotiation callback)
    react to state changes without polling.
    """

    def __init__(self) -> None:
        self._ledger: Optional[StateLedger] = None
        self._revision: int = 0
        self._lock = threading.RLock()
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
        Create and store a fresh, validated ledger (loop_count=0).

        `transferable_units`, `air_freight_available`, and `air_freight_capacity_units` seed
        the mitigation feasibility signals; `delay_days` seeds the observed shipment slip that
        drives the dynamic financial exposure. Returns the newly created ledger.
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
        """Return a deep copy of the current ledger; mutating it never affects stored state."""
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

        `patch` is a nested dict keyed by top-level section, e.g.
        `{"metrics": {"revenue_at_risk_usd": 4200.0}}`. The merged result is re-validated
        through the full Pydantic model, so any value violating a type/Literal constraint
        raises and the mutation is rejected — the deterministic State Mutation Layer enforcement point.
        """
        with self._lock:
            if self._ledger is None:
                raise RuntimeError("No active incident. Call init_incident() first.")

            merged = self._deep_merge(self._ledger.model_dump(), patch)
            validated = StateLedger.model_validate(merged)
            self._ledger = validated
            self._revision += 1
            self._notify()
            return self.snapshot()

    def increment_loop(self) -> StateLedger:
        """Increment loop_count (the UPDATE STATE LEDGER node)."""
        current = self.snapshot()
        return self.mutate(
            {"metadata": {"loop_count": current.metadata.loop_count + 1}}
        )

    # ------------------------------------------------------------------ #
    # Raw log sink — never enters the ledger
    # ------------------------------------------------------------------ #
    @staticmethod
    def append_raw_log(source: str, raw_payload: str) -> None:
        """
        Append verbose/unstructured tool output to the isolated execution log.

        Both `source` and `raw_payload` may carry adversarial content (LLM output, vendor
        replies, blocked injection blobs), so embedded control characters are stripped before
        writing — this is the single write boundary, so every call site is protected against
        log forging and terminal corruption. The trailing "\\n" is the only newline in a record.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        safe_source = _LOG_CONTROL_CHARS_RE.sub(" ", str(source))
        safe_payload = _LOG_CONTROL_CHARS_RE.sub(" ", str(raw_payload))
        with RAW_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"[{timestamp}] [{safe_source}] {safe_payload}\n")

    @staticmethod
    async def append_raw_log_async(source: str, raw_payload: str) -> None:
        """Non-blocking variant of `append_raw_log` for use inside async coroutines.

        Offloads the blocking file write to a worker thread so the event loop stays free —
        important inside the concurrent negotiation sub-graphs, where a blocking write would
        serialize otherwise-concurrent vendor LLM calls. Reuses `append_raw_log`, so the same
        control-char sanitization applies.
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
                result[key] = LedgerStore._deep_merge(
                    cast(Dict[str, Any], existing), cast(Dict[str, Any], value)
                )
            else:
                result[key] = value
        return result


# Module-level singleton the rest of the system shares by default.
STORE = LedgerStore()


__all__ = ["LedgerStore", "STORE", "RAW_LOG_PATH"]
