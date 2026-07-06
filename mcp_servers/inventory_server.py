"""
Inventory / WMS MCP server — warehouse management tools.

Exposes the stock-visibility tool:
  - query_inventory -> plant stock balances, consumption speeds, safety thresholds

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
from tools.mock_data import INVENTORY_DB as _INVENTORY_DB

ToolResult = dict[str, str | int | float | bool | list[str] | dict[str, int]]


def _query_inventory(sku_id: str) -> Dict[str, Any]:
    """Pull plant stock balances, consumption speeds, and minimum safety thresholds."""
    record = _INVENTORY_DB.get(sku_id)
    if record is None:
        LedgerStore.append_raw_log("query_inventory", f"MISS sku_id={sku_id}")
        return {"found": False, "sku_id": sku_id}

    LedgerStore.append_raw_log(
        "query_inventory", f"plant_balances={record['plant_balances']}"
    )
    return {
        "found": True,
        "sku_id": record["sku_id"],
        "plant_balances": record["plant_balances"],
        "daily_consumption_units": record["daily_consumption_units"],
        "safety_stock_threshold": record["safety_stock_threshold"],
        "inventory_days_remaining": record["inventory_days_remaining"],
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
inventory_app = _FastMCP("oscar-inventory-wms") if _FastMCP is not None else None


def query_inventory(sku_id: str) -> ToolResult:
    """MCP tool: plant stock balances, consumption speeds, safety thresholds."""
    return _query_inventory(sku_id)


if inventory_app is not None:
    inventory_app.tool()(query_inventory)


def main() -> None:
    """Entry point: launch the Inventory/WMS MCP server over stdio transport."""
    if not MCP_AVAILABLE or inventory_app is None:
        raise RuntimeError("The 'mcp' package is not installed. Run `pip install mcp`.")
    inventory_app.run()


if __name__ == "__main__":
    main()


__all__ = ["query_inventory", "inventory_app", "MCP_AVAILABLE", "ToolResult"]

