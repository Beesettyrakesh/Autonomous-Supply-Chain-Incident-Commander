"""
Backward-compatibility facade for the MCP Observation Tools.

The tools now live in three category servers under the `mcp_servers/` package
(ERP / Inventory-WMS / Logistics-TMS). This module re-exports their public functions so any
caller that historically did `import mcp_server` / `mcp_server.query_erp(...)` keeps working.

New code should prefer importing from `mcp_servers` (the orchestrator wires its tool registry
from the individual servers directly).
"""

from __future__ import annotations

from mcp_servers import (
    query_erp,
    query_inventory,
    query_shipment_tracking,
    extract_contract_rules,
    query_alternate_suppliers,
    erp_app,
    inventory_app,
    logistics_app,
    MCP_AVAILABLE,
)



def main() -> None:
    """Category MCP servers run as separate processes in production. Run each server module
    directly, e.g. `python -m mcp_servers.erp_server`.
    """

    raise RuntimeError(
        "Run a specific category server, e.g. `python -m mcp_servers.erp_server` "
        "(also: mcp_servers.inventory_server, mcp_servers.logistics_server)."
    )


__all__ = [
    "query_erp",
    "query_inventory",
    "query_shipment_tracking",
    "extract_contract_rules",
    "query_alternate_suppliers",
    "erp_app",
    "inventory_app",
    "logistics_app",
    "MCP_AVAILABLE",
]

