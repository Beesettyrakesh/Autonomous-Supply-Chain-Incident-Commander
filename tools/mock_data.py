"""
Deterministic mock enterprise datasets for the MCP Observation Tools.

Seeded around the target incident: a critical shipment delay on SKU-99 from Supplier A
causing a downstream shutdown risk at Plant 2. Provides the data the Incident Commander
reasons over for the three mitigation strategies:
    - ALT_SUPPLIER      -> approved alternate vendors (SUP-B, SUP-C) can supply SKU-99.
    - INTERNAL_TRANSFER -> PLANT-1 holds surplus stock transferable to PLANT-2.
    - AIR_FREIGHT       -> expedite the existing delayed PO by air.

These are plain Python dicts; the MCP tools read from them and return strict primitives.
"""

from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------- #
# ERP: purchase orders, vendor master, base lead-time frameworks.
# ---------------------------------------------------------------------------- #
ERP_DB: Dict[str, Dict[str, Any]] = {
    "SKU-99": {
        "sku_id": "SKU-99",
        "primary_supplier_id": "SUP-A",
        "active_contract_id": "CTR-4471",
        "current_purchase_order_id": "PO-88123",
        "base_lead_time_days": 14,
        "unit_cost_usd": 42.50,
        "vendor_master": {"name": "Supplier A", "rating": "APPROVED", "region": "APAC"},
        # Two approved alternates with distinct cost bases, so a concurrent negotiation
        # yields distinct floors and a meaningful lowest-price winner.
        "alternate_suppliers": [
            {
                "supplier_id": "SUP-B",
                "name": "Supplier B",
                "rating": "APPROVED",
                "region": "EU",
                "unit_cost_usd": 46.75,       # pricier -> higher walk-away floor.
                "quoted_lead_time_days": 8,
                "min_order_qty": 250,
            },
            {
                "supplier_id": "SUP-C",
                "name": "Supplier C",
                "rating": "APPROVED",
                "region": "NA",
                "unit_cost_usd": 44.00,       # cheapest AND fastest -> loss-minimizing winner.
                "quoted_lead_time_days": 6,
                "min_order_qty": 300,
            },

        ],
    }
}

# ---------------------------------------------------------------------------- #
# Inventory: plant balances, consumption speeds, safety thresholds, transfer feasibility.
# ---------------------------------------------------------------------------- #
INVENTORY_DB: Dict[str, Dict[str, Any]] = {
    "SKU-99": {
        "sku_id": "SKU-99",
        "plant_balances": {"PLANT-1": 1800, "PLANT-2": 320},
        "daily_consumption_units": {"PLANT-1": 150, "PLANT-2": 160},
        "safety_stock_threshold": {"PLANT-1": 600, "PLANT-2": 500},
        "inventory_days_remaining": 2,  # Plant 2 is critically short (~2 days).
        # PLANT-1 surplus above its own safety stock, transferable to PLANT-2.
        "transfer_options": [
            {
                "from_plant": "PLANT-1",
                "to_plant": "PLANT-2",
                "transferable_units": 350,
                "transit_days": 1,
                "transfer_cost_usd": 850.0,
            }
        ],

    }
}

# ---------------------------------------------------------------------------- #
# Shipment tracking: transit status + updated ETA telemetry (the disruption trigger).
# ---------------------------------------------------------------------------- #
SHIPMENT_DB: Dict[str, Dict[str, Any]] = {
    "PO-88123": {
        "po_id": "PO-88123",
        "carrier": "OceanFreight Co",
        "origin": "Shanghai, CN",
        "destination": "Plant 2, DE",
        "original_eta": "2026-07-04",
        "updated_eta": "2026-07-13",  # 9-day slip -> the disruption trigger.
        "delay_days": 9,
        "status": "DELAYED",
        "last_known_coordinates": {"lat": 1.29, "lon": 103.85},
        # Expedite-by-air feasibility for the existing PO (AIR_FREIGHT strategy).
        "air_freight_option": {
            "available": True,
            "expedited_transit_days": 2,
            "surcharge_usd": 3200.0,
        },
    }
}


__all__ = ["ERP_DB", "INVENTORY_DB", "SHIPMENT_DB"]
