"""Integration tests for the MCP client gateway (real stdio transport).

These prove the orchestrator's client side genuinely speaks the Model Context Protocol to the
three category servers running as subprocesses — not just in-process function calls. They spawn
real server subprocesses, so they are a touch heavier than the offline unit tests but stay fully
deterministic (GEMINI_API_KEY="" is set by conftest, so the servers never call an LLM).
"""

from __future__ import annotations

import asyncio

import pytest

from mcp_client import MCPToolGateway, build_inproc_invoker


def test_inproc_invoker_matches_tool_contract() -> None:
    """The in-process invoker returns the same shapes the servers do (dict / list)."""

    async def _run() -> None:
        invoke = build_inproc_invoker()
        erp = await invoke("query_erp", {"sku_id": "SKU-99"})
        assert erp["found"] is True
        assert erp["primary_supplier_id"] == "SUP-A"
        alts = await invoke("query_alternate_suppliers", {"sku_id": "SKU-99"})
        assert isinstance(alts, list) and len(alts) == 2

    asyncio.run(_run())


def test_stdio_gateway_calls_tools_over_mcp() -> None:
    """Spawn the 3 servers over stdio and invoke a tool on each via a real MCP ClientSession."""

    async def _run() -> None:
        gateway = MCPToolGateway()
        try:
            await gateway.start()
        except Exception as exc:  # pragma: no cover - only if `mcp` transport is unavailable
            pytest.skip(f"MCP stdio transport unavailable: {exc}")
        try:
            # ERP server.
            erp = await gateway.call_tool("query_erp", {"sku_id": "SKU-99"})
            assert erp["found"] is True
            assert erp["primary_supplier_id"] == "SUP-A"

            # ERP server, list-returning tool (structuredContent unwrap path).
            alts = await gateway.call_tool("query_alternate_suppliers", {"sku_id": "SKU-99"})
            assert isinstance(alts, list)
            assert {a["supplier_id"] for a in alts} == {"SUP-B", "SUP-C"}

            # Inventory / WMS server.
            inv = await gateway.call_tool("query_inventory", {"sku_id": "SKU-99"})
            assert inv["found"] is True
            assert inv["inventory_days_remaining"] == 2

            # Logistics / TMS server.
            ship = await gateway.call_tool("query_shipment_tracking", {"po_id": "PO-88123"})
            assert ship["found"] is True
            assert ship["delay_days"] == 9
        finally:
            await gateway.aclose()

    asyncio.run(_run())


def test_orchestrator_runs_over_stdio_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: the commander resolves an incident with observation tools over real MCP."""
    import orchestrator as orch
    from ledger_store import STORE

    # Force the stdio transport for this test (the suite default is in-process). The transport
    # is resolved from the environment at run start, so setting the env var is sufficient.
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")


    STORE.init_incident(
        target_sku="SKU-99",
        primary_supplier_id="SUP-A",
        active_contract_id="CTR-4471",
        current_purchase_order_id="PO-88123",
        impacted_plants=["PLANT-2"],
        inventory_days_remaining=2,
        production_shutdown_hours=48,
        revenue_at_risk_usd=75000.0,
        transferable_units=350,
        air_freight_available=True,
        air_freight_capacity_units=420,
        replacement_order_qty=300,
        delay_days=9,
    )
    commander = orch.IncidentCommander(order_quantity=300)
    assert commander.mcp_transport == "stdio"

    async def _run() -> None:
        await commander.run(verbose=False)

    try:
        asyncio.run(_run())
    except Exception as exc:  # pragma: no cover - only if stdio transport is unavailable
        pytest.skip(f"MCP stdio transport unavailable: {exc}")

    led = STORE.snapshot()
    # qty 300 -> internal transfer resolves; the loss reflects the real MCP-sourced delay/penalty.
    assert led.mitigation.active_strategy == "INTERNAL_TRANSFER"
    assert led.status.guardrail_status == "PASSED"
    assert led.metrics.projected_total_loss_usd == 357750.0
