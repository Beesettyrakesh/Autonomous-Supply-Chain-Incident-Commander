"""
MCP server ecosystem for OSCAR.

The Observation Tools are split across three category servers, mirroring how a real enterprise
exposes purpose-specific systems:
  - erp_app        (ERP)            -> query_erp, query_alternate_suppliers, extract_contract_rules
  - inventory_app  (Inventory/WMS)  -> query_inventory
  - logistics_app  (Logistics/TMS)  -> query_shipment_tracking

This package re-exports each server's public tool functions so callers can reach them from one
import surface while the servers stay categorically separated. (The orchestrator imports the
submodules directly for explicit per-server wiring.)
"""

from mcp_servers.erp_server import (
    query_erp,
    query_alternate_suppliers,
    extract_contract_rules,
    erp_app,
    MCP_AVAILABLE,
)
from mcp_servers.inventory_server import query_inventory, inventory_app
from mcp_servers.logistics_server import query_shipment_tracking, logistics_app

__all__ = [
    "query_erp",
    "query_alternate_suppliers",
    "extract_contract_rules",
    "query_inventory",
    "query_shipment_tracking",
    "erp_app",
    "inventory_app",
    "logistics_app",
    "MCP_AVAILABLE",
]
