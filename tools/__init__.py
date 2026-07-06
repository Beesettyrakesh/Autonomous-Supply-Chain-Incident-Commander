"""
Observation-tool support package.

Houses the deterministic mock enterprise datasets (ERP / inventory / shipment) that the
MCP Observation Tools query, keeping the tool transport layer decoupled from the data.
"""

from tools.mock_data import ERP_DB, INVENTORY_DB, SHIPMENT_DB

__all__ = ["ERP_DB", "INVENTORY_DB", "SHIPMENT_DB"]
