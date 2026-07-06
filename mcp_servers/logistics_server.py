"""
Logistics / TMS MCP server — transportation management tools.

Exposes the shipment-visibility tool:
  - query_shipment_tracking -> transit coordinates, updated ETA telemetry, delay days

Tool contract (shared across all category servers):
- Accept strictly-typed scalar inputs.
- Return strictly-typed JSON-serializable dicts (never raw prose or markdown).
- Push verbose/raw source payloads to `incident_execution.log`, not into the return value.

Built on `mcp` (FastMCP). If `mcp` is not installed the module still imports cleanly and the
underlying `_query_*` implementation remains directly callable/testable.
"""

from __future__ import annotations

from typing import Any, Dict

from ledger_store import LedgerStore
from tools.mock_data import SHIPMENT_DB as _SHIPMENT_DB

ToolResult = dict[str, str | int | float | bool | list[str] | dict[str, int]]


def _query_shipment_tracking(po_id: str) -> Dict[str, Any]:
    """Pull transit coordinates and updated ETA telemetry for a purchase order."""
    record = _SHIPMENT_DB.get(po_id)
    if record is None:
        LedgerStore.append_raw_log("query_shipment_tracking", f"MISS po_id={po_id}")
        return {"found": False, "po_id": po_id}

    LedgerStore.append_raw_log(
        "query_shipment_tracking",
        f"coords={record['last_known_coordinates']} carrier={record['carrier']}",
    )
    return {
        "found": True,
        "po_id": record["po_id"],
        "status": record["status"],
        "original_eta": record["original_eta"],
        "updated_eta": record["updated_eta"],
        "delay_days": record["delay_days"],
        "destination": record["destination"],
    }


# ---------------------------------------------------------------------------- #
# MCP server wiring (FastMCP). Applied only if the `mcp` package is available.
# ---------------------------------------------------------------------------- #
try:
    from mcp.server.fastmcp import FastMCP

    _FastMCP = FastMCP
except ImportError:  # pragma: no cover - graceful degradation when mcp not installed
    _FastMCP = None

MCP_AVAILABLE = _FastMCP is not None
logistics_app = _FastMCP("oscar-logistics-tms") if _FastMCP is not None else None


def query_shipment_tracking(po_id: str) -> ToolResult:
    """MCP tool: transit coordinates and updated ETA telemetry."""
    return _query_shipment_tracking(po_id)


if logistics_app is not None:
    logistics_app.tool()(query_shipment_tracking)


def main() -> None:
    """Entry point: launch the Logistics/TMS MCP server over stdio transport."""
    if not MCP_AVAILABLE or logistics_app is None:
        raise RuntimeError("The 'mcp' package is not installed. Run `pip install mcp`.")
    logistics_app.run()


if __name__ == "__main__":
    main()


__all__ = ["query_shipment_tracking", "logistics_app", "MCP_AVAILABLE", "ToolResult"]

