"""
MCP Server exposing the Observation Tools.

These I/O-bound functions query environmental reality (ERP, inventory, shipment tracking,
contract rules, alternate suppliers). Surfacing them through a dedicated MCP server keeps
the orchestrator decoupled from the underlying (mock) enterprise systems.

Tool contract:
- Accept strictly-typed scalar inputs.
- Return strictly-typed JSON-serializable dicts (never raw prose or markdown).
- Push verbose/raw source payloads to `incident_execution.log`, not into the return value.

Built on `mcp` (FastMCP). If `mcp` is not installed the module still imports cleanly and the
underlying `_query_*` implementations remain directly callable/testable. Run: `python mcp_server.py`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

from ledger_store import LedgerStore

# Mock datasets live in the tools/ package, keeping tool transport separate from mock data.
from tools.mock_data import ERP_DB as _ERP_DB
from tools.mock_data import INVENTORY_DB as _INVENTORY_DB
from tools.mock_data import SHIPMENT_DB as _SHIPMENT_DB

# Directory the observation tools resolve local mock artifacts (e.g. contract files) from.
_MODULE_DIR = Path(__file__).parent

# Return contract for the primary observation tools: scalars, a list of strings, or a shallow
# {str: int} map (e.g. plant balances) — no `Any`.
ToolResult = dict[str, str | int | float | bool | list[str] | dict[str, int]]

# Return contract for alternate-supplier discovery: a list of flat vendor records.
SupplierRecord = dict[str, str | int | float]
SupplierListResult = list[SupplierRecord]


# ---------------------------------------------------------------------------- #
# Core implementations (framework-agnostic, unit-testable). Each pushes verbose context
# to the raw log and returns strict primitives only.
# ---------------------------------------------------------------------------- #
def _query_erp(sku_id: str) -> Dict[str, Any]:
    """Pull active PO records, vendor master files, and base lead-time frameworks."""
    record = _ERP_DB.get(sku_id)
    if record is None:
        LedgerStore.append_raw_log("query_erp", f"MISS sku_id={sku_id}")
        return {"found": False, "sku_id": sku_id}

    LedgerStore.append_raw_log("query_erp", f"vendor_master={record['vendor_master']}")
    return {
        "found": True,
        "sku_id": record["sku_id"],
        "primary_supplier_id": record["primary_supplier_id"],
        "active_contract_id": record["active_contract_id"],
        "current_purchase_order_id": record["current_purchase_order_id"],
        "base_lead_time_days": record["base_lead_time_days"],
        "unit_cost_usd": record["unit_cost_usd"],
    }


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


def _query_alternate_suppliers(sku_id: str) -> list[Dict[str, Any]]:
    """Discover approved alternate vendors that can fulfil this SKU (the ALT_SUPPLIER path).

    Returns a list of flat vendor records with only the scalar fields the negotiation needs.
    """
    record = _ERP_DB.get(sku_id) or {}
    alternates = record.get("alternate_suppliers", [])
    LedgerStore.append_raw_log(
        "query_alternate_suppliers",
        f"sku_id={sku_id} count={len(alternates)}",
    )
    return [
        {
            "supplier_id": str(a["supplier_id"]),
            "name": str(a.get("name", a["supplier_id"])),
            "unit_cost_usd": float(a["unit_cost_usd"]),
            "quoted_lead_time_days": int(a.get("quoted_lead_time_days", 0)),
            "min_order_qty": int(a.get("min_order_qty", 0)),
        }
        for a in alternates
    ]


# Matches "3.0% per diem", "3 % per-diem", etc. Captures the numeric percentage value.
_PENALTY_CLAUSE_RE = re.compile(
    r"penalt(?:y|ies)\s+shall\s+accrue\s+at\s+([0-9]+(?:\.[0-9]+)?)\s*%\s*per[\s-]?diem",
    re.IGNORECASE,
)


def _extract_contract_rules(contract_id: str) -> Dict[str, Any]:
    """Read `contract_{contract_id}.txt` and extract the per-diem late-delivery penalty rate.

    The full contract body is verbose, so only the parsed numeric primitive is returned; the
    raw text is pushed to the isolated execution log.
    """
    contract_path = _MODULE_DIR / f"contract_{contract_id}.txt"
    if not contract_path.exists():
        LedgerStore.append_raw_log(
            "extract_contract_rules", f"MISS contract_id={contract_id} (no file)"
        )
        return {"found": False, "contract_id": contract_id}

    raw_text = contract_path.read_text(encoding="utf-8")
    LedgerStore.append_raw_log(
        "extract_contract_rules",
        f"contract_id={contract_id} bytes={len(raw_text)} raw={raw_text!r}",
    )

    match = _PENALTY_CLAUSE_RE.search(raw_text)
    if match is None:
        return {"found": True, "contract_id": contract_id, "contracted_penalty_rate": 0.0}

    # Convert the captured percentage (e.g. "3.0") to a fractional rate (0.03).
    contracted_penalty_rate = round(float(match.group(1)) / 100.0, 4)
    return {
        "found": True,
        "contract_id": contract_id,
        "contracted_penalty_rate": contracted_penalty_rate,
    }


# ---------------------------------------------------------------------------- #
# MCP Server wiring (FastMCP). Applied only if the `mcp` package is available so the module
# remains importable in constrained/test environments.
# ---------------------------------------------------------------------------- #
try:
    from mcp.server.fastmcp import FastMCP

    _FastMCP = FastMCP
except ImportError:  # pragma: no cover - graceful degradation when mcp not installed
    _FastMCP = None

MCP_AVAILABLE = _FastMCP is not None
mcp_server = _FastMCP("supply-chain-observation-tools") if _FastMCP is not None else None


# Public tool functions are always defined so callers/tests work with or without `mcp`.
def query_erp(sku_id: str) -> ToolResult:
    """MCP tool: active purchase order records, vendor master, base lead-time."""
    return _query_erp(sku_id)


def query_inventory(sku_id: str) -> ToolResult:
    """MCP tool: plant stock balances, consumption speeds, safety thresholds."""
    return _query_inventory(sku_id)


def query_shipment_tracking(po_id: str) -> ToolResult:
    """MCP tool: transit coordinates and updated ETA telemetry."""
    return _query_shipment_tracking(po_id)


def extract_contract_rules(contract_id: str) -> ToolResult:
    """MCP tool: parse the contract file and return the per-diem penalty rate primitive."""
    return _extract_contract_rules(contract_id)


def query_alternate_suppliers(sku_id: str) -> SupplierListResult:
    """MCP tool: approved alternate vendors that can fulfil the SKU (ALT_SUPPLIER path)."""
    return _query_alternate_suppliers(sku_id)


# Register the tools with the MCP server only when the framework is present.
if mcp_server is not None:
    mcp_server.tool()(query_erp)
    mcp_server.tool()(query_inventory)
    mcp_server.tool()(query_shipment_tracking)
    mcp_server.tool()(extract_contract_rules)
    mcp_server.tool()(query_alternate_suppliers)


def main() -> None:
    """Entry point: launch the MCP server over stdio transport."""
    if not MCP_AVAILABLE or mcp_server is None:
        raise RuntimeError(
            "The 'mcp' package is not installed. Run `pip install mcp` to serve tools."
        )
    mcp_server.run()


if __name__ == "__main__":
    main()


__all__ = [
    "query_erp",
    "query_inventory",
    "query_shipment_tracking",
    "extract_contract_rules",
    "query_alternate_suppliers",
    "mcp_server",
    "MCP_AVAILABLE",
]
