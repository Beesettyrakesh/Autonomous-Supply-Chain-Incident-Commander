"""
MCP client gateway — the orchestrator's real client side of the Model Context Protocol.

The three category servers (`mcp_servers/erp_server.py`, `inventory_server.py`,
`logistics_server.py`) are genuine FastMCP servers. This module is the matching CLIENT: it
launches each server as its own subprocess and speaks MCP to it over stdio, exactly as an
enterprise orchestrator would reach purpose-specific tool services.

Two transports are supported so the same tool surface works everywhere:
  - "stdio"  : REAL MCP — each server runs as a subprocess; tools are invoked via an MCP
               `ClientSession.call_tool(...)`. Used for the live demo.
  - "inproc" : the tool functions are imported and called directly in-process. Deterministic
               and dependency-free, used by the offline test suite and CI.

The orchestrator asks this gateway for an async tool invoker and never cares which transport
is active — the observation-tool call sites are identical.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union, cast

# Async callable the orchestrator uses to invoke a single observation tool.
ToolInvoker = Callable[[str, Dict[str, Any]], Awaitable[Any]]

_REPO_ROOT = Path(__file__).parent

# Which category server owns each observation tool (mirrors the 3-server split).
TOOL_SERVER_MAP: Dict[str, str] = {
    "query_erp": "erp",
    "query_alternate_suppliers": "erp",
    "extract_contract_rules": "erp",
    "query_inventory": "inventory",
    "query_shipment_tracking": "logistics",
}

# Runnable module path for each category server (launched via `python -m ...`).
SERVER_MODULES: Dict[str, str] = {
    "erp": "mcp_servers.erp_server",
    "inventory": "mcp_servers.inventory_server",
    "logistics": "mcp_servers.logistics_server",
}


def _parse_tool_result(result: Any) -> Any:
    """Normalize an MCP CallToolResult back into the plain dict/list the orchestrator expects.

    FastMCP returns dict-shaped tool outputs in `structuredContent` directly, but wraps a bare
    list return under a single `{"result": [...]}` key. We unwrap that so the client yields the
    exact same Python value the in-process function would have returned. Falls back to parsing
    the JSON text content if structured content is unavailable.
    """
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
            return structured["result"]
        return structured

    # Fallback: reconstruct from the text content blocks.
    content = getattr(result, "content", None) or []
    texts = [getattr(block, "text", "") for block in content if getattr(block, "text", None)]
    if not texts:
        return {"error": "empty MCP tool result"}
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except json.JSONDecodeError:
            return {"error": texts[0]}
    # Multiple text blocks -> a list of parsed items.
    parsed: List[Any] = []
    for t in texts:
        try:
            parsed.append(json.loads(t))
        except json.JSONDecodeError:
            parsed.append(t)
    return parsed


class MCPToolGateway:
    """Manages the three MCP server subprocesses and routes tool calls to the right session.

    Lifecycle: `await start()` spawns all three servers and opens a `ClientSession` per server;
    `await aclose()` tears everything down. `call_tool(name, args)` dispatches to the session
    that owns the tool and returns the parsed primitive result.
    """

    def __init__(
        self,
        *,
        python_executable: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self._python = python_executable or sys.executable
        # Inherit the parent environment so path/CWD resolution works; silence the benign
        # runpy re-import warning for a clean demo transcript.
        self._env = dict(env or os.environ)
        self._env.setdefault("PYTHONWARNINGS", "ignore")
        self._stack: Optional[AsyncExitStack] = None
        self._sessions: Dict[str, Any] = {}
        self._started = False

    async def start(self) -> None:
        """Spawn each category server over stdio and initialize an MCP session for it."""
        if self._started:
            return
        # Imported lazily so importing this module never hard-requires `mcp` (offline path).
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stack = AsyncExitStack()
        try:
            for name, module in SERVER_MODULES.items():
                params = StdioServerParameters(
                    command=self._python,
                    args=["-m", module],
                    env=self._env,
                    cwd=str(_REPO_ROOT),
                )
                read, write = await self._stack.enter_async_context(stdio_client(params))
                session = await self._stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._sessions[name] = session
            self._started = True
        except BaseException:
            # Ensure any partially-opened subprocesses are cleaned up on a failed start.
            await self.aclose()
            raise

    async def call_tool(self, tool: str, args: Dict[str, Any]) -> Any:
        """Invoke `tool` on its owning server over MCP and return the parsed result."""
        if not self._started:
            raise RuntimeError("MCPToolGateway.start() must be awaited before call_tool().")
        server = TOOL_SERVER_MAP.get(tool)
        if server is None:
            return {"error": f"tool '{tool}' is not served by any MCP server"}
        session = self._sessions[server]
        result = await session.call_tool(tool, args)
        if getattr(result, "isError", False):
            return {"error": f"MCP tool '{tool}' reported an error"}
        return _parse_tool_result(result)

    async def aclose(self) -> None:
        """Close all sessions and terminate the server subprocesses."""
        self._started = False
        self._sessions = {}
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None


def build_inproc_invoker() -> ToolInvoker:
    """Return an async invoker that calls the tool functions directly in-process.

    Used by the offline/test path: no subprocesses, fully deterministic, and it still presents
    the same async `(tool, args) -> result` contract the orchestrator uses for MCP.
    """
    import mcp_servers.erp_server as erp
    import mcp_servers.inventory_server as inventory
    import mcp_servers.logistics_server as logistics

    registry: Dict[str, Callable[..., Any]] = {
        "query_erp": erp.query_erp,
        "query_alternate_suppliers": erp.query_alternate_suppliers,
        "extract_contract_rules": erp.extract_contract_rules,
        "query_inventory": inventory.query_inventory,
        "query_shipment_tracking": logistics.query_shipment_tracking,
    }

    async def _invoke(tool: str, args: Dict[str, Any]) -> Any:
        fn = registry.get(tool)
        if fn is None:
            return {"error": f"unknown observation tool '{tool}'"}
        return fn(**args)

    return _invoke


__all__ = [
    "MCPToolGateway",
    "build_inproc_invoker",
    "ToolInvoker",
    "TOOL_SERVER_MAP",
    "SERVER_MODULES",
]
