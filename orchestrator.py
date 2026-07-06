"""
The central Incident Commander agent — the LLM reasoning core.

Implements the closed ReAct (Reason + Act) loop that drives the system. Each cycle:
    1. Read the serialized JSON of the State Ledger from `ledger_store.STORE`.
    2. Reason: the LLM emits a `Thought:` line identifying data gaps / options.
    3. Act: the LLM emits a tool call matching a registered MCP Observation Tool or
       Decision Helper.
    4. The tool is dispatched deterministically; its parsed primitives are committed back
       to the ledger via the State Mutation Layer (`STORE.mutate`).
    5. `loop_count` is incremented; the circuit breaker forces a stop past MAX_LOOPS.

The API key is read from the environment (`GEMINI_API_KEY`) and the model from `GEMINI_MODEL`.
If the SDK or key is unavailable, the module still imports cleanly and falls back to a
deterministic offline planner so the loop stays testable end-to-end without a live model.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union, cast


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional; env vars still work directly
    pass

from pydantic import ValidationError

from ledger_store import STORE, LedgerStore
# MCP Observation Tools are split across three category servers (ERP / Inventory-WMS /
# Logistics-TMS). The commander reaches them as an MCP CLIENT through `mcp_client`: in the
# live path each server runs as its own subprocess and tools are invoked over the MCP protocol
# (stdio); the offline/test path calls the same tool functions in-process for determinism.
import mcp_servers.erp_server as erp
import mcp_servers.inventory_server as inventory
import mcp_servers.logistics_server as logistics
import decision_helpers
from mcp_client import MCPToolGateway, build_inproc_invoker, ToolInvoker



from negotiation_agent import run_supplier_negotiation, TermsDict
from guardrails import (
    check_spend_authority,
    sanitize_write_payload,
    InjectionAttemptError,
    SpendAuthorityResult,
    SPEND_AUTHORITY_LIMIT_USD,
)
from llm_utils import generate_with_retry, LLMUnavailableError

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
# Circuit-breaker bound mirrored from the schema (loop_count le=11, escalate past 10).
MAX_LOOPS = 10

# MCP transport for the Observation Tools (read live from the environment at run start, so a
# UI/CLI that sets it just before launching the loop is honoured):
#   "stdio"  -> REAL MCP: each category server runs as a subprocess and tools are invoked over
#               the MCP protocol (used for the live demo).
#   "inproc" -> the same tool functions are called directly in-process (deterministic; the
#               default for the offline test suite and CI).
def _resolve_mcp_transport() -> str:
    """Return the active MCP transport ('stdio' or 'inproc') from the environment."""
    return "stdio" if os.environ.get("MCP_TRANSPORT", "inproc").strip().lower() == "stdio" else "inproc"




# Order quantity for the mitigation purchase — the per-incident variable that drives total
# spend (unit price * qty) against the fixed spend-authority limit. Env-configurable so a demo
# can flip between the auto-approve path (low qty) and the HITL path (high qty).
ORDER_QUANTITY = int(os.environ.get("ORDER_QUANTITY", "500"))

# HITL decision channel: 'cli' prompts the operator at the terminal; 'auto_approve' /
# 'auto_reject' are non-interactive defaults for tests/CI. The dashboard injects its own
# callable directly, bypassing this switch.
HITL_MODE = os.environ.get("HITL_MODE", "cli")

# Human-in-the-loop decision provider: given the breach details, return True to APPROVE the
# over-limit spend or False to REJECT. May be sync or async — `_enforce_spend_guardrail`
# awaits either transparently, so the dashboard can inject a sync callback and the CLI stays
# non-blocking.
HumanDecisionFn = Callable[[SpendAuthorityResult], Union[bool, Awaitable[bool]]]

# Observational event sink (same pattern as LedgerStore.on_mutation): the orchestrator narrates
# each ReAct step as a (kind, message, data) tuple for the presentation layer. It never affects
# control flow, and any exception in the sink is swallowed so telemetry can't break the agent.
EventFn = Callable[[str, str, Dict[str, Any]], None]


async def _default_human_decision(result: SpendAuthorityResult) -> bool:
    """Default HITL provider driven by HITL_MODE (cli | auto_approve | auto_reject).

    Returns True to APPROVE the over-limit purchase, False to REJECT. The auto_* modes make
    deterministic choices for offline tests/CI. The CLI mode reads operator input via
    `asyncio.to_thread(input, ...)` so the blocking terminal prompt runs on a worker thread
    and never stalls the event loop.
    """
    if HITL_MODE == "auto_approve":
        return True
    if HITL_MODE == "auto_reject":
        return False
    # Interactive CLI approval gate.
    prompt = (
        "\n" + "=" * 68 + "\n"
        "[HUMAN-IN-THE-LOOP] Spend exceeds the agent's delegated authority.\n"
        f"  Supplier : {result.supplier_id}\n"
        f"  Spend    : ${result.spend_usd:,.2f}\n"
        f"  Limit    : ${result.limit_usd:,.2f}\n"
        f"  Reason   : {result.reason}\n"
        + "=" * 68 + "\n"
        "Approve this purchase? [y/N]: "
    )
    try:
        # Offload blocking input() to a worker thread so it doesn't block the event loop.
        answer = (await asyncio.to_thread(input, prompt)).strip().lower()
    except EOFError:  # non-interactive stdin -> safe default is to REJECT
        return False
    return answer in ("y", "yes")


# ---------------------------------------------------------------------------- #
# Tool registry — the only actions the orchestrator may select each turn.
# Maps the tool name the LLM emits to a concrete Python callable.
# ---------------------------------------------------------------------------- #
ToolFn = Callable[..., Any]

TOOL_REGISTRY: Dict[str, ToolFn] = {
    # ERP server (purchasing / sourcing).
    "query_erp": erp.query_erp,
    "extract_contract_rules": erp.extract_contract_rules,
    # Inventory / WMS server.
    "query_inventory": inventory.query_inventory,
    # Logistics / TMS server.
    "query_shipment_tracking": logistics.query_shipment_tracking,
    # Decision Helpers (deterministic math).
    "simulate_finance": decision_helpers.simulate_finance,
    "score_strategy": decision_helpers.score_strategy,
    "policy_check": decision_helpers.policy_check,
}



# Observation tools served over MCP (the client routes these to the category servers). The
# Decision Helpers (simulate_finance / score_strategy / policy_check) are deterministic local
# math and are intentionally NOT MCP tools — they run in-process.
OBSERVATION_TOOLS = frozenset(
    {"query_erp", "query_inventory", "query_shipment_tracking", "extract_contract_rules"}
)


# Allowed mitigation strategies — must match the Literal set in schema.MitigationState.
# Published to the LLM so it never invents an out-of-schema value (e.g. "EXPEDITE").
VALID_STRATEGIES = ("ALT_SUPPLIER", "INTERNAL_TRANSFER", "AIR_FREIGHT")


# Virtual (non-registry) actions the orchestrator handles itself. `commit_strategy` is a
# selection decision; `DONE` is the terminal signal.
CONTROL_ACTIONS = ("commit_strategy", "DONE")
KNOWN_TOOLS = frozenset(TOOL_REGISTRY) | frozenset(CONTROL_ACTIONS)


# Tool contracts injected into the system prompt so the LLM emits valid tool-calling syntax.
TOOL_CATALOG = f"""
AVAILABLE TOOLS (select exactly one per turn):
- query_erp(sku_id: str)                    -> PO records, vendor master, base lead-time
- query_inventory(sku_id: str)              -> plant balances, consumption, safety stock
- query_shipment_tracking(po_id: str)       -> transit status, updated ETA, delay_days
- extract_contract_rules(contract_id: str)  -> contracted_penalty_rate from the contract
- simulate_finance(delay_days: int)         -> daily penalty + projected total loss
      pass the delay observed via query_shipment_tracking (call that FIRST so the
      loss reflects the real slip; omitting it falls back to the ledger's delay_days)

- score_strategy(strategy_type: str)        -> cost/time/risk + composite score (EVALUATE only)
      strategy_type MUST be exactly one of: {", ".join(VALID_STRATEGIES)}
      (do NOT invent other values such as "EXPEDITE" — use "AIR_FREIGHT" to expedite)
      NOTE: scoring only records a score; it does NOT select the strategy.
- commit_strategy(strategy_type: str)       -> COMMIT the chosen strategy as active
      strategy_type MUST be exactly one of: {", ".join(VALID_STRATEGIES)}
- policy_check(supplier_id: str, spend_amount: float) -> compliance bool
"""

SYSTEM_PROMPT = f"""
You are the AUTONOMOUS SUPPLY CHAIN INCIDENT COMMANDER, an enterprise employee agent.
Your sole mission: resolve critical procurement incidents before they become catastrophic
financial and operational disruptions.

STRICT OPERATING PROTOCOL (ReAct):
1. At the START of EVERY cycle you are given the serialized JSON of the Structured State
   Ledger. You MUST inspect it to identify information gaps or the best mitigation path.
2. You MUST first output your reasoning on a single line prefixed exactly with 'Thought:'.
   Explain the data gap you are filling or the option you are evaluating.
3. Then you MUST select your Next Best Action by emitting a tool call on a single line
   prefixed exactly with 'Action:' using strict JSON, e.g.:
      Action: {{"tool": "query_shipment_tracking", "args": {{"po_id": "PO-88123"}}}}
4. You may ONLY call the tools listed below. You MUST NOT invent tools or arguments.
5. You are PHYSICALLY FORBIDDEN from computing financial impact or strategy scores in your
   own words — you MUST delegate that to simulate_finance / score_strategy.
6. When the incident is fully assessed and a mitigation strategy is chosen, output:
      Action: {{"tool": "DONE", "args": {{}}}}

SEQUENCING RULES (follow this logical order):
- Extract contract rules (extract_contract_rules) to populate contracted_penalty_rate
  BEFORE calling simulate_finance — otherwise the penalty computes as 0 and is useless.
- Only call score_strategy AFTER you understand the delay and financial exposure.
- Choose the single best mitigation strategy, then emit DONE.

FEASIBILITY RULES (a strategy must be POSSIBLE before you may commit it — this is separate
from its score; you MUST NOT commit an infeasible option even if it scores highest). Reason
about these like a procurement manager, in plain business terms — state the REAL reason an
option is or isn't executable this incident, not just a true/false flag:
- INTERNAL_TRANSFER: only works if the sister site's transferable surplus can actually cover
  the required replacement quantity (metrics.transferable_units >= metrics.replacement_order_qty).
  If the available surplus is smaller than the shortfall, an internal transfer CANNOT close the
  gap this incident — so explain it that way and move to the next viable option.
- AIR_FREIGHT: only works if the lane is open (metrics.air_freight_available is true) AND the
  carrier's finite air cargo capacity can cover the required quantity
  (metrics.air_freight_capacity_units >= metrics.replacement_order_qty). Aircraft hold volume is
  limited, so a large order can EXCEED the available air capacity — in that case air freight
  cannot be executed regardless of how attractive its speed score is; say so plainly, citing the
  capacity vs the required quantity. (Do NOT justify skipping it on unit cost.)
- ALT_SUPPLIER: the approved alternate-vendor pool is always a viable sourcing path, so when
  the internal and expedite routes are closed, source from an alternate supplier and negotiate.
This mirrors real procurement escalation: use internal stock if it suffices, else expedite by
air if there is enough air capacity, else source from an approved alternate supplier. Which one
is feasible EMERGES from the required quantity — you are not told which to pick.

{TOOL_CATALOG}

ONE-SHOT FORMAT EXAMPLE (this exact two-line structure is REQUIRED every turn):
Thought: The contract penalty rate is still 0.0, so I must parse the contract before I can simulate finance accurately.
Action: {{"tool": "extract_contract_rules", "args": {{"contract_id": "CTR-4471"}}}}

Respond with ONLY the 'Thought:' line and the 'Action:' line. No extra commentary.

"""


# ---------------------------------------------------------------------------- #
# Gemini client bootstrap.
# ---------------------------------------------------------------------------- #
def _init_genai_client() -> Optional[Any]:
    """Build the Google Gen AI client from GEMINI_API_KEY.

    Returns None if the SDK is not installed or no key is present, enabling the offline fallback.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai  # unified google-genai SDK
    except ImportError:
        return None
    return genai.Client(api_key=api_key)


class IncidentCommander:
    """Central orchestrator, wired to the `STORE` singleton for reads/mutations."""

    def __init__(
        self,
        store: Optional[LedgerStore] = None,
        *,
        order_quantity: int = ORDER_QUANTITY,
        spend_authority_limit_usd: float = SPEND_AUTHORITY_LIMIT_USD,
        human_decision: Optional[HumanDecisionFn] = None,
        on_event: Optional[EventFn] = None,
    ) -> None:
        self.store = store or STORE
        self._client = _init_genai_client()
        # True when a live LLM core is available; else we use the offline planner.
        self.llm_enabled = self._client is not None
        # Order quantity drives total spend against the fixed authority limit — flip it to
        # demo the auto-approve vs HITL paths.
        self.order_quantity = order_quantity
        self.spend_authority_limit_usd = spend_authority_limit_usd
        # HITL decision provider; defaults to the env-driven CLI/auto provider (the dashboard
        # injects its own callable).
        self.human_decision: HumanDecisionFn = human_decision or _default_human_decision
        # Observational narration sink (optional); behaviour-neutral.
        self.on_event: Optional[EventFn] = on_event
        # MCP client gateway (stdio transport) — created on demand in `run()` when
        # MCP_TRANSPORT=="stdio"; None on the in-process path.
        self._mcp_gateway: Optional[MCPToolGateway] = None
        # Async invoker for the observation tools. Defaults to the in-process invoker so a
        # commander used directly (e.g. a unit test calling `_dispatch_tool`) always works; the
        # stdio gateway invoker is installed for the duration of `run()` when selected.
        self._tool_invoker: ToolInvoker = build_inproc_invoker()

    @property
    def mcp_transport(self) -> str:
        """Active MCP transport for the observation tools ('stdio' or 'inproc')."""
        return _resolve_mcp_transport()



    def _emit(self, kind: str, message: str, **data: Any) -> None:
        """Fire the observational event sink (if any). Never affects control flow.

        `kind` is a machine tag the UI maps to an icon (thought | action | observation |
        negotiation | guardrail | hitl | resolution | error); `message` is a human-readable
        line; `data` carries structured extras. Any exception in the sink is swallowed.
        """
        if self.on_event is None:
            return
        try:
            self.on_event(kind, message, data)
        except Exception:  # pragma: no cover - a broken UI sink must not crash the agent
            pass

    @staticmethod
    def _humanize_observation(tool: str, result: Any) -> str:
        """Translate a tool's raw result into a data-rich, plain-English line for the UI."""
        if not isinstance(result, dict):
            return str(result)
        r = cast(Dict[str, Any], result)
        if "error" in r:
            return str(r["error"])

        if tool == "extract_contract_rules":
            rate = r.get("contracted_penalty_rate", 0.0)
            return f"Contract {r.get('contract_id', '')} parsed → late penalty {rate * 100:.1f}%/day"
        if tool == "query_shipment_tracking":
            return (
                f"Shipment {r.get('po_id', '')} is {r.get('status', '?')} "
                f"{r.get('delay_days', 0)} days (updated ETA {r.get('updated_eta', '?')})"
            )
        if tool == "query_erp":
            return (
                f"ERP: primary supplier {r.get('primary_supplier_id', '?')} @ "
                f"${r.get('unit_cost_usd', 0):,.2f}/unit, base lead {r.get('base_lead_time_days', 0)}d"
            )
        if tool == "query_inventory":
            return f"Inventory: {r.get('inventory_days_remaining', 0)} days of cover remaining"
        if tool == "simulate_finance":
            return (
                f"Projected total loss ${r.get('projected_total_loss_usd', 0):,.0f} "
                f"(daily penalty ${r.get('daily_penalty_usd', 0):,.0f})"
            )
        if tool == "score_strategy":
            return (
                f"Scored {r.get('strategy_type', '?')} → composite {r.get('composite_score', 0)} "
                f"(cost {r.get('cost_score', 0)} · time {r.get('time_score', 0)} · risk {r.get('risk_score', 0)})"
            )
        if tool == "commit_strategy":
            return f"Committed strategy: {r.get('committed_strategy', '?')}"
        return json.dumps(r)

    def _summarize_resolution(self, ledger: Any) -> str:
        """Build a plain-English RESOLUTION summary for the UI, derived from the final ledger.

        States HOW the incident was resolved (or WHY escalated), the strategy chosen, the
        vendor/terms/spend when a purchase was made, and the projected loss averted (or that
        remains unmitigated). Exactly one leading status glyph (✅ / 🚨) is kept.
        """
        mit = ledger.mitigation
        metrics = ledger.metrics
        status = ledger.status
        strat = mit.active_strategy
        loss = metrics.projected_total_loss_usd

        # Escalated: guardrail breached and not overridden -> no mitigation executed.
        if status.guardrail_status == "BREACHED":
            return (
                f"🚨 Incident ESCALATED to a human — {status.escalation_reason or 'guardrail breached'}. "
                f"No purchase order was placed; the projected loss of ${loss:,.0f} remains "
                "unmitigated and awaits manual handling."
            )

        # Resolved via an alternate-supplier purchase (negotiated PO).
        if strat == "ALT_SUPPLIER":
            spend = mit.agreed_unit_price_usd * self.order_quantity
            override = (
                " (human-approved over-limit spend)"
                if status.escalation_reason == "human_approved_over_limit_spend"
                else ""
            )
            return (
                f"✅ Incident RESOLVED via ALT_SUPPLIER — PO placed with "
                f"{mit.agreed_supplier_id} for {self.order_quantity} units @ "
                f"${mit.agreed_unit_price_usd:,.2f}/unit (${spend:,.0f} total, "
                f"{mit.agreed_lead_time_days}-day lead){override}. "
                f"Projected loss of ${loss:,.0f} averted."
            )

        # Resolved via internal stock transfer (no spend).
        if strat == "INTERNAL_TRANSFER":
            return (
                f"✅ Incident RESOLVED via INTERNAL_TRANSFER — {metrics.transferable_units} "
                f"surplus units re-routed to cover the {metrics.replacement_order_qty}-unit "
                f"shortfall at $0 external spend. Projected loss of ${loss:,.0f} averted."
            )

        # Resolved via air-freight expedite.
        if strat == "AIR_FREIGHT":
            return (
                f"✅ Incident RESOLVED via AIR_FREIGHT — the delayed shipment is being "
                f"expedited by air to protect production. Projected loss of ${loss:,.0f} averted."
            )

        # Fallback (no strategy recorded but not breached).
        return (
            f"✅ Incident resolved (strategy: {strat}). "
            f"Projected loss of ${loss:,.0f} addressed."
        )

    # ------------------------------------------------------------------ #
    # Reasoning: produce a (thought, action) decision for the current ledger.
    # ------------------------------------------------------------------ #
    async def _reason(self, ledger_json: str, scratchpad: str) -> Dict[str, Any]:
        """Ask the reasoning core for the next Thought + Action given ledger + history."""
        if self.llm_enabled and self._client is not None:
            return await self._reason_with_llm(ledger_json, scratchpad)
        return self._reason_offline(ledger_json)

    async def _reason_with_llm(self, ledger_json: str, scratchpad: str) -> Dict[str, Any]:
        """Invoke Gemini (async) and parse the 'Thought:' / 'Action:' response.

        The `scratchpad` carries the running ReAct history (prior Thought / Action /
        Observation turns) so the model remembers what it already did — the Observation step
        that prevents blind, repeated tool calls.
        """
        from google.genai import types  # type: ignore

        history_block = (
            f"REACT HISTORY (your prior turns and their observations):\n{scratchpad}\n\n"
            if scratchpad.strip()
            else "REACT HISTORY: (none yet — this is the first turn)\n\n"
        )
        contents = (
            f"CURRENT STATE LEDGER (JSON):\n{ledger_json}\n\n"
            f"{history_block}"
            "Do NOT repeat an Action you have already performed if its Observation is "
            "already available above. Select your next Thought and Action."
        )
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,  # deterministic reasoning for reproducible demos
        )
        response = await self._generate_with_retry(contents, config)
        return self._parse_react_text(response.text or "")

    async def _generate_with_retry(self, contents: str, config: Any) -> Any:
        """Call Gemini via the shared resilient helper so both agents share one retry policy."""
        return await generate_with_retry(
            self._client,
            model=GEMINI_MODEL,
            contents=contents,
            config=config,
            source="orchestrator",
            log=LedgerStore.append_raw_log,
        )

    def _reason_offline(self, ledger_json: str) -> Dict[str, Any]:
        """Deterministic fallback planner used when no live model is configured.

        Walks the same information-gathering order a well-behaved LLM would follow, driven by
        which ledger fields are still unpopulated, so the closed loop is fully exercisable in
        CI / smoke tests without an API key.
        """
        ledger: Dict[str, Any] = json.loads(ledger_json)
        ctx: Dict[str, Any] = ledger["context"]
        mit: Dict[str, Any] = ledger["mitigation"]
        metrics: Dict[str, Any] = ledger["metrics"]
        scores: Dict[str, Any] = mit.get("strategy_scores", {})

        # 1) Parse the contract to learn the penalty rate.
        if ctx.get("contracted_penalty_rate", 0.0) == 0.0:
            return {
                "thought": "No contracted penalty rate yet; I must parse the active contract.",
                "action": {
                    "tool": "extract_contract_rules",
                    "args": {"contract_id": ctx["active_contract_id"]},
                },
            }
        # 2) Quantify financial exposure. Pass no explicit delay_days so simulate_finance reads
        # the observed slip from the ledger (metrics.delay_days) — this makes the projected
        # loss dynamic: change the delay and the exposure recomputes.
        if metrics.get("projected_total_loss_usd", 0.0) == 0.0:
            return {
                "thought": "Penalty rate known; simulate the financial impact of the observed delay.",
                "action": {"tool": "simulate_finance", "args": {}},
            }

        # 3) Evaluate (score) the candidate strategies — recording only, not selecting.
        for candidate in VALID_STRATEGIES:
            if candidate not in scores:
                return {
                    "thought": f"Evaluate {candidate} so I can compare options before committing.",
                    "action": {"tool": "score_strategy", "args": {"strategy_type": candidate}},
                }
        # 4) All scored: commit the best FEASIBLE strategy (committing is terminal, so no
        # separate DONE turn). The feasibility gate is separate from the scores: an
        # infeasible-but-high-scoring option is excluded so the agent organically escalates
        # to ALT_SUPPLIER. `pool` never goes empty (ALT_SUPPLIER is always feasible).
        feasible = {s: v for s, v in scores.items() if self._is_strategy_feasible(s, metrics)}
        pool = feasible or scores
        best = max(pool, key=lambda k: pool[k])
        thought = self._narrate_commit_reasoning(best, feasible, metrics)
        return {
            "thought": thought,
            "action": {"tool": "commit_strategy", "args": {"strategy_type": best}},
        }

    def _narrate_commit_reasoning(
        self, best: str, feasible: Dict[str, Any], metrics: Dict[str, Any]
    ) -> str:
        """Compose the offline planner's final 'Thought' from the real feasibility numbers.

        Reads like a procurement manager stating the true reason each closed option was ruled
        out, then why it commits the chosen one — mirroring the live SYSTEM_PROMPT wording.
        Purely narration; never changes the choice.
        """
        transferable = int(metrics.get("transferable_units", 0))
        required = int(metrics.get("replacement_order_qty") or self.order_quantity)
        air_available = bool(metrics.get("air_freight_available", True))
        air_capacity = int(metrics.get("air_freight_capacity_units", 0))

        reasons: List[str] = []
        if "INTERNAL_TRANSFER" not in feasible:
            reasons.append(
                f"the sister site's transferable surplus ({transferable} units) can't cover the "
                f"{required}-unit shortfall, so an internal transfer won't close the gap"
            )
        if "AIR_FREIGHT" not in feasible:
            # Distinguish WHY air is closed: lane grounded vs finite capacity can't fit the order.
            if not air_available:
                reasons.append(
                    "the carrier has no air capacity on this lane, so air freight can't be executed"
                )
            else:
                reasons.append(
                    f"the available air cargo capacity ({air_capacity} units) can't fit the "
                    f"{required}-unit order, so air freight can't be executed at this volume"
                )

        if best == "ALT_SUPPLIER":
            lead = "With the internal and expedite routes closed, " if reasons else ""
            because = (" — " + "; ".join(reasons) + ".") if reasons else "."
            return (
                f"{lead}the approved alternate-vendor pool remains a viable sourcing path{because} "
                "I'll source from an alternate supplier and negotiate terms."
            )
        if best == "INTERNAL_TRANSFER":
            return (
                f"The sister site's transferable surplus ({transferable} units) fully covers the "
                f"{required}-unit shortfall, so an internal stock transfer resolves this at no external "
                "spend — committing INTERNAL_TRANSFER."
            )
        if best == "AIR_FREIGHT":
            surplus_note = (
                f"the internal surplus ({transferable} units) can't cover the {required}-unit order, but "
                if "INTERNAL_TRANSFER" not in feasible
                else ""
            )
            return (
                f"{surplus_note}the air cargo capacity ({air_capacity} units) can expedite the full "
                f"{required}-unit order, which best protects production against the downtime exposure "
                "— committing AIR_FREIGHT."
            )
        return f"Committing {best} as the best feasible mitigation for this incident."

    def _is_strategy_feasible(self, strategy: str, metrics: Dict[str, Any]) -> bool:
        """Deterministic feasibility gate — can this strategy be executed this incident?

        Distinct from the desirability score: gating selection on feasibility is how the agent
        escalates the procurement ladder (internal stock → expedite → alternate supplier)
        without fudging the scores.

        - INTERNAL_TRANSFER: feasible only if PLANT-1's transferable surplus covers the full
          replacement order quantity.
        - AIR_FREIGHT: feasible only if the lane is open AND finite air capacity covers the qty.
        - ALT_SUPPLIER: always feasible (approved alternate vendor pool).

        The required quantity is read from `metrics.replacement_order_qty` — the same value the
        LLM sees — so the model, offline planner, and this backstop all compare one number,
        falling back to `self.order_quantity` only if the ledger field is unset.
        """
        if strategy == "INTERNAL_TRANSFER":
            required = int(metrics.get("replacement_order_qty") or self.order_quantity)
            return int(metrics.get("transferable_units", 0)) >= required
        if strategy == "AIR_FREIGHT":
            required = int(metrics.get("replacement_order_qty") or self.order_quantity)
            capacity = int(metrics.get("air_freight_capacity_units", 0))
            return bool(metrics.get("air_freight_available", True)) and capacity >= required
        return True

    @staticmethod
    def _extract_json_object(blob: str) -> Optional[Dict[str, Any]]:
        """Best-effort extraction of a single JSON object from messy LLM text.

        Strips markdown code fences, then scans from the first '{' with brace-depth counting to
        the matching '}' so exactly one balanced object is isolated (tolerating trailing prose /
        pretty-printing). Returns the parsed dict, or None if no valid object is found.
        """
        # 1) Strip markdown code fences (```json ... ``` or ``` ... ```).
        cleaned = re.sub(r"```(?:json)?", "", blob, flags=re.IGNORECASE).replace("```", "")

        # 2) Brace-matching scan from the first '{'.
        start = cleaned.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        return None
                    return cast(Dict[str, Any], parsed) if isinstance(parsed, dict) else None
        return None

    @classmethod
    def _parse_react_text(cls, text: str) -> Dict[str, Any]:
        """Extract the 'Thought:' string and the JSON 'Action:' object from raw output.

        Thinking models emit multi-line reasoning after 'Thought:' (captured via DOTALL) and
        often wrap the Action JSON in code fences. We isolate the payload after 'Action:' and
        parse it with a brace-matching, fence-tolerant extractor so valid tool calls aren't
        silently downgraded to DONE.
        """
        # Capture multi-line thought up to the Action token (or end of text).
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\n\s*Action:|\Z)", text, re.DOTALL | re.IGNORECASE
        )
        thought = thought_match.group(1).strip() if thought_match else ""

        # Grab everything after the first 'Action:' token, then extract one JSON object.
        action: Dict[str, Any] = {"tool": "DONE", "args": {}}
        action_split = re.split(r"Action:\s*", text, maxsplit=1, flags=re.IGNORECASE)
        if len(action_split) == 2:
            parsed = cls._extract_json_object(action_split[1])
            if isinstance(parsed, dict) and "tool" in parsed:
                parsed.setdefault("args", {})
                action = parsed
            # else: malformed/absent tool syntax -> safe deterministic DONE.
        return {"thought": thought, "action": action}

    # ------------------------------------------------------------------ #
    # Acting: dispatch the selected tool and translate output into a ledger patch.
    # ------------------------------------------------------------------ #
    async def _dispatch_tool(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered tool, injecting the ledger snapshot where required.

        Observation tools are invoked through the active MCP transport (`self._tool_invoker`):
        over the MCP protocol to a category-server subprocess under the stdio transport, or
        directly in-process under the deterministic transport. Decision Helpers run locally.
        Validates arguments defensively BEFORE running the tool so a schema-noncompliant LLM
        output becomes a recoverable error Observation instead of a downstream crash.
        """

        # Security: sanitize every dispatched tool's args (defense-in-depth). A hit aborts with
        # a recoverable error Observation and the loop continues safely.
        try:
            sanitize_write_payload(args)
        except InjectionAttemptError as exc:
            LedgerStore.append_raw_log("security", f"INJECTION_BLOCKED tool={tool} args={args} err={exc}")
            return {"result": {"error": f"tool call blocked by security layer: {exc}"}}

        # Reject out-of-schema strategy names up front (score_strategy and commit_strategy).
        if tool in ("score_strategy", "commit_strategy"):
            strat = str(args.get("strategy_type", ""))
            if strat not in VALID_STRATEGIES:
                return {
                    "result": {
                        "error": (
                            f"invalid strategy_type '{strat}'. "
                            f"Choose exactly one of: {', '.join(VALID_STRATEGIES)}."
                        )
                    }
                }

        # commit_strategy simply confirms the chosen strategy (args already sanitized above).
        if tool == "commit_strategy":
            strat = str(args.get("strategy_type", ""))
            # Feasibility backstop: enforce feasibility in code so a live model that ignores the
            # prompt rule can't drive an impossible action. A blocked commit becomes a
            # recoverable error Observation, so the agent re-selects a feasible option — this is
            # what organically routes a depleted-stock scenario to ALT_SUPPLIER.
            metrics = self.store.snapshot_dict().get("metrics", {})
            if not self._is_strategy_feasible(strat, metrics):
                LedgerStore.append_raw_log(
                    "orchestrator", f"INFEASIBLE_COMMIT_BLOCKED strategy={strat} metrics={metrics}"
                )
                return {
                    "result": {
                        "error": (
                            f"strategy '{strat}' is INFEASIBLE for this incident "
                            f"(transferable_units={metrics.get('transferable_units')}, "
                            f"air_freight_available={metrics.get('air_freight_available')}, "
                            f"order_quantity={self.order_quantity}). "
                            "Commit a feasible strategy instead (ALT_SUPPLIER is always feasible)."
                        )
                    }
                }
            return {"result": {"committed_strategy": strat}}

        # Observation tools go through the active MCP transport (stdio subprocess or in-process
        # invoker). This is the real MCP client boundary.
        if tool in OBSERVATION_TOOLS:
            try:
                result = await self._tool_invoker(tool, args)
            except Exception as exc:  # transport/tool failure -> recoverable error Observation
                LedgerStore.append_raw_log("orchestrator", f"MCP_TOOL_ERROR tool={tool} args={args} err={exc!r}")
                return {"result": {"error": f"MCP tool '{tool}' failed: {exc}"}}
            return {"result": result}

        # Decision Helpers run locally (deterministic math, not MCP tools).
        fn = TOOL_REGISTRY[tool]
        if tool in ("simulate_finance", "score_strategy"):
            args = {**args, "state_ledger_snapshot": self.store.snapshot_dict()}
        # Resilience: guard the call so a bad signature (e.g. a missing arg) becomes a
        # recoverable error Observation instead of crashing the ReAct loop.
        try:
            result = fn(**args)
        except TypeError as exc:
            LedgerStore.append_raw_log("orchestrator", f"BAD_TOOL_ARGS tool={tool} args={args} err={exc}")
            return {"result": {"error": f"invalid arguments for '{tool}': {exc}"}}
        return {"result": result}


    def _mutation_for(self, tool: str, output: Dict[str, Any]) -> Dict[str, Any]:
        """Map a tool's parsed output to a validated State Ledger patch (Mutation Layer)."""
        raw = output.get("result")
        if not isinstance(raw, dict):
            return {}
        result = cast(Dict[str, Any], raw)

        if tool == "extract_contract_rules" and result.get("found"):
            return {"context": {"contracted_penalty_rate": result["contracted_penalty_rate"]}}
        if tool == "query_shipment_tracking" and result.get("found"):
            # Record only the observed delay. Do NOT overwrite production_shutdown_hours — a
            # shipment delay is not the same as shutdown hours; simulate_finance remains the sole
            # authority on downtime math (delay beyond inventory_days_remaining).
            delay_days = int(result.get("delay_days", 0))
            if delay_days > 0:
                return {"metrics": {"delay_days": delay_days}}
            return {}

        if tool == "simulate_finance":
            # Persist the computed financial primitives so the ledger captures the exposure.
            return {
                "metrics": {
                    "daily_penalty_usd": result["daily_penalty_usd"],
                    "projected_total_loss_usd": result["projected_total_loss_usd"],
                }
            }
        if tool == "score_strategy":
            # Evaluate only: record the score without selecting it. The deep-merge preserves
            # previously-scored sibling strategies, so a leaf patch is sufficient.
            strat = str(result["strategy_type"])
            return {"mitigation": {"strategy_scores": {strat: result["composite_score"]}}}
        if tool == "commit_strategy":
            # Select: set the active strategy after comparing the recorded scores.
            strat = str(result.get("committed_strategy", ""))
            if strat in VALID_STRATEGIES:
                return {"mitigation": {"active_strategy": strat}}
            return {}
        # Observation tools that only inform reasoning need no mutation this turn.
        return {}

    # ------------------------------------------------------------------ #
    # The closed ReAct loop.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _render_scratchpad(trace: List[Dict[str, Any]]) -> str:
        """Render the accumulated ReAct history into a compact prompt-ready transcript."""
        lines: List[str] = []
        for i, turn in enumerate(trace):
            lines.append(f"Turn {i}:")
            lines.append(f"  Thought: {turn.get('thought', '')}")
            lines.append(f"  Action: {turn.get('tool')} {turn.get('args', {})}")
            # The Observation is the actual tool output fed back to the model — what turns
            # blind repetition into grounded reasoning.
            lines.append(f"  Observation: {json.dumps(turn.get('output'))}")
        return "\n".join(lines)

    async def _maybe_negotiate(self, verbose: bool) -> None:
        """After an ALT_SUPPLIER commit, negotiate with all alternate suppliers concurrently
        (asyncio.gather) and select the lowest-price winner.

        The entire multi-vendor negotiation is ONE orchestrator action — this method awaits the
        sub-graphs but does NOT call increment_loop(). The orchestrator holds the master status
        lock: it writes negotiation_status=IN_PROGRESS once, runs the suppliers with
        write_status=False (so they never race on the shared field), then writes the single
        final SUCCESS/FAILED after evaluating results.
        """
        snapshot = self.store.snapshot()
        # Discover alternate vendors through the MCP layer (never a direct DB import) so the
        # orchestrator stays decoupled from the ERP data source.
        alts = cast(
            List[Dict[str, Any]],
            erp.query_alternate_suppliers(snapshot.context.target_sku),
        )
        incident_id = str(snapshot.metadata.id)

        # Buyer's willingness-to-pay ceiling is our own economics: the primary supplier's unit
        # cost plus an acceptable premium (sourced from the ERP server), not the vendor's price.
        primary = cast(Dict[str, Any], erp.query_erp(snapshot.context.target_sku))

        primary_unit_cost = float(primary.get("unit_cost_usd", 42.5)) if primary.get("found") else 42.5
        ACCEPTABLE_PREMIUM = 1.10  # willing to pay up to 10% over primary cost to resolve.
        buyer_ceiling = round(primary_unit_cost * ACCEPTABLE_PREMIUM, 2)

        if not alts:
            # No alternate vendors to negotiate with -> clean FAILED, no crash.
            self.store.mutate({"mitigation": {"negotiation_status": "FAILED"}})
            if verbose:
                print("[SUB-GRAPH] No alternate suppliers available; negotiation FAILED.")
            return

        # Orchestrator takes the master status lock before spawning concurrent negotiations.
        self.store.mutate({"mitigation": {"negotiation_status": "IN_PROGRESS"}})
        vendor_ids = ", ".join(str(a["supplier_id"]) for a in alts)
        self._emit(
            "negotiation",
            f"Negotiating concurrently with {len(alts)} alternate suppliers ({vendor_ids})…",
            vendors=vendor_ids,
        )
        if verbose:
            print(f"[SUB-GRAPH] Negotiating {len(alts)} suppliers concurrently (asyncio.gather) ...")

        # One negotiation coroutine per alternate supplier:
        #   - buyer ceiling  = our willingness-to-pay (shared across vendors).
        #   - supplier floor = that vendor's true unit cost (distinct per vendor, so quotes differ).
        async def negotiate_one(alt: Dict[str, Any]) -> TermsDict:
            vendor_floor = float(alt["unit_cost_usd"])  # vendor won't go below its own cost
            return await run_supplier_negotiation(
                incident_id,
                {
                    "unit_price_ceiling_usd": buyer_ceiling,
                    "required_lead_time_days": int(alt.get("quoted_lead_time_days", 6)),
                    "qty": self.order_quantity,
                },
                supplier_id=str(alt["supplier_id"]),
                floor_price=vendor_floor,
                lead_time_days=int(alt.get("quoted_lead_time_days", 6)),
                store=self.store,
                write_status=False,  # orchestrator owns the shared status field
            )

        # return_exceptions=True so a single vendor failure doesn't abort the gather; failures
        # come back as Exception objects that we log and drop, proceeding with what succeeded.
        raw_results: List[Any] = await asyncio.gather(
            *(negotiate_one(a) for a in alts), return_exceptions=True
        )
        results: List[TermsDict] = []
        for alt, r in zip(alts, raw_results):
            if isinstance(r, BaseException):
                LedgerStore.append_raw_log(
                    "orchestrator",
                    f"negotiation ERROR supplier={alt.get('supplier_id')} err={r!r}",
                )
                continue
            results.append(cast(TermsDict, r))

        # Keep only successful quotes.
        successful = [r for r in results if r.get("negotiation_outcome_status") == "SUCCESS"]

        # Never call min() on an empty list.
        if len(successful) == 0:
            self.store.mutate({"mitigation": {"negotiation_status": "FAILED"}})
            LedgerStore.append_raw_log(
                "orchestrator", f"negotiation ALL-FAILED incident={incident_id} results={results}"
            )
            if verbose:
                print("[SUB-GRAPH] All supplier negotiations failed; negotiation FAILED.")
            return

        # Select the lowest agreed unit price among successful vendors.
        best = min(successful, key=lambda r: float(r["agreed_unit_price_usd"]))

        LedgerStore.append_raw_log(
            "orchestrator",
            f"negotiation WINNER incident={incident_id} supplier={best.get('agreed_supplier_id')} "
            f"price={best.get('agreed_unit_price_usd')} from candidates={[r.get('agreed_supplier_id') for r in successful]}",
        )

        # Orchestrator writes the single final status + winning primitive terms.
        self.store.mutate(
            {
                "mitigation": {
                    "negotiation_status": "SUCCESS",
                    "agreed_supplier_id": str(best["agreed_supplier_id"]),
                    "agreed_unit_price_usd": float(best["agreed_unit_price_usd"]),
                    "agreed_lead_time_days": int(best["agreed_lead_time_days"]),
                }
            }
        )
        # Negotiation transcript (UI narration): a clean SEQUENTIAL reconstruction from the real
        # agreed numbers. The live sub-graph bargains concurrently; this is purely observational
        # and never changes the outcome. Emitted with role tags for chat-bubble rendering.
        best_supplier = str(best["agreed_supplier_id"])
        best_price = float(best["agreed_unit_price_usd"])
        best_lead = int(best["agreed_lead_time_days"])
        opening_offer = round(buyer_ceiling * 0.90, 2)  # mirrors the sub-graph's opening step

        self._emit(
            "negotiation",
            f"Opening the bidding for {self.order_quantity} units at ${opening_offer:,.2f}/unit "
            f"(walk-away ceiling ${buyer_ceiling:,.2f}).",
            role="system",
        )
        # Sort by price so the transcript reads cheapest-first and ends on the winner.
        for r in sorted(successful, key=lambda x: float(x["agreed_unit_price_usd"])):
            sid = str(r["agreed_supplier_id"])
            price = float(r["agreed_unit_price_usd"])
            lead = int(r["agreed_lead_time_days"])
            self._emit(
                "negotiation",
                f"We can offer ${opening_offer:,.2f}/unit for {self.order_quantity} units on a rush order.",
                role="buyer",
                supplier=sid,
            )
            self._emit(
                "negotiation",
                f"The best I can do is ${price:,.2f}/unit with a {lead}-day lead time.",
                role="supplier",
                supplier=sid,
            )

        # Buyer closes with the lowest-cost vendor.
        self._emit(
            "negotiation",
            f"Agreed at ${best_price:,.2f}/unit with {best_supplier} — the lowest-cost quote.",
            role="buyer",
            supplier=best_supplier,
        )
        self._emit(
            "negotiation",
            f"Best quote secured: {best_supplier} @ ${best_price:,.2f}/unit, {best_lead}-day lead.",
            role="system",
            supplier=best_supplier,
            unit_price=best_price,
        )
        if verbose:
            print(f"[SUB-GRAPH] Best quote: {best}")

        # Financial spend-authority guardrail: before the deal is treated as resolved, check
        # whether total spend is within the delegated authority. Over-limit spend hard-forks to
        # a human-in-the-loop approve/reject decision — the LLM has no say in this barrier.
        await self._enforce_spend_guardrail(best, verbose)

    async def _enforce_spend_guardrail(self, best: TermsDict, verbose: bool) -> None:
        """Run the spend-authority guardrail on a won deal and drive the HITL decision.

        Outcomes written to the ledger:
        - within authority         -> guardrail PASSED, deal stands.
        - over authority, APPROVED -> human override: guardrail PASSED, deal stands.
        - over authority, REJECTED -> guardrail BREACHED, active_strategy cleared (no PO),
          negotiation_status FAILED, escalation_reason set — incident returned to a human.
        """
        supplier_id = str(best["agreed_supplier_id"])
        unit_price = float(best["agreed_unit_price_usd"])
        check = check_spend_authority(
            supplier_id,
            unit_price,
            self.order_quantity,
            limit_usd=self.spend_authority_limit_usd,
        )
        LedgerStore.append_raw_log("guardrail", f"spend_authority {check}")

        if check.within_authority:
            # Within delegated authority -> auto-approved, no human needed.
            self.store.mutate({"status": {"guardrail_status": "PASSED"}})
            self._emit(
                "guardrail",
                f"🛡️ Spend ${check.spend_usd:,.0f} ≤ ${check.limit_usd:,.0f} authority → "
                "auto-approved (within delegated authority)",
                passed=True,
            )
            if verbose:
                print(f"[GUARDRAIL] {check.reason}")
            return

        # Over authority -> record the breach (so a watcher can render it), then ask a human.
        self.store.mutate(
            {"status": {"guardrail_status": "BREACHED", "escalation_reason": check.reason}}
        )
        self._emit(
            "guardrail",
            f"🛑 Spend ${check.spend_usd:,.0f} EXCEEDS ${check.limit_usd:,.0f} authority "
            f"(supplier {supplier_id}) → escalating to a human for approval",
            passed=False,
        )
        self._emit(
            "hitl",
            f"Awaiting human decision on ${check.spend_usd:,.0f} over-limit spend…",
            spend=check.spend_usd,
            limit=check.limit_usd,
        )

        if verbose:
            print(f"[GUARDRAIL] BREACHED — {check.reason}")

        # Obtain the verdict. The provider may be sync or async — await it transparently.
        decision = self.human_decision(check)
        if inspect.isawaitable(decision):
            decision = await decision
        approved = bool(decision)
        LedgerStore.append_raw_log(
            "guardrail",
            f"HITL decision approved={approved} supplier={supplier_id} spend={check.spend_usd}",
        )

        if approved:
            # Human override authorizes the over-limit purchase; the deal proceeds.
            self.store.mutate(
                {
                    "status": {
                        "guardrail_status": "PASSED",
                        "escalation_reason": "human_approved_over_limit_spend",
                    }
                }
            )
            self._emit(
                "hitl",
                f"Human APPROVED — PO authorized with {supplier_id} for ${check.spend_usd:,.0f}",
                approved=True,
            )
            if verbose:
                print(f"[HITL] APPROVED — PO authorized with {supplier_id} for ${check.spend_usd:,.2f}.")
            return

        # Human rejected: cancel the purchase, keep BREACHED, escalate. Clearing active_strategy
        # signals no mitigation executed, and we NULL the negotiated primitives too — otherwise
        # they'd strand a phantom PO for a cancelled purchase in the single source of truth.
        self.store.mutate(
            {
                "mitigation": {
                    "active_strategy": "NONE",
                    "negotiation_status": "FAILED",
                    "agreed_supplier_id": None,
                    "agreed_unit_price_usd": 0.0,
                    "agreed_lead_time_days": 0,
                },
                "status": {
                    "guardrail_status": "BREACHED",
                    "escalation_reason": "human_rejected_over_limit_spend",
                },
            }
        )

        self._emit(
            "hitl",
            f"Human REJECTED — no PO placed for {supplier_id}; the over-limit "
            f"${check.spend_usd:,.0f} spend was denied and the incident is escalated "
            "for manual handling.",
            approved=False,
        )

        if verbose:
            print(f"[HITL] REJECTED — no PO placed; incident escalated for manual handling.")

    async def run(self, max_loops: int = MAX_LOOPS, verbose: bool = True) -> List[Dict[str, Any]]:
        """Execute the autonomous reasoning loop until DONE, goal achieved, or the circuit
        breaker trips (loop_count > max_loops). Returns the ordered trace of turns.

        A running ReAct scratchpad is maintained and injected into each reasoning call, giving
        the model memory of prior observations so it converges instead of repeating actions.
        Async so it can `await` the negotiation sub-graph.
        """
        trace: List[Dict[str, Any]] = []
        await self._start_transport(verbose)
        try:
            return await self._run_loop(trace, max_loops, verbose)
        finally:
            await self._stop_transport()

    async def _start_transport(self, verbose: bool) -> None:
        """Bring up the MCP client gateway when the stdio transport is selected.

        Spawns the three category servers as subprocesses and installs the gateway's async
        invoker. If startup fails, we fall back to the in-process invoker so the run still
        completes — the observation tools produce identical results either way.
        """
        if self.mcp_transport != "stdio":
            return
        gateway = MCPToolGateway()
        try:
            await gateway.start()
        except Exception as exc:  # pragma: no cover - defensive: degrade to in-process
            LedgerStore.append_raw_log("orchestrator", f"MCP_STDIO_START_FAILED err={exc!r}; using in-process tools")
            if verbose:
                print(f"[MCP] stdio transport unavailable ({exc}); falling back to in-process tools.")
            await gateway.aclose()
            return
        self._mcp_gateway = gateway
        self._tool_invoker = gateway.call_tool
        if verbose:
            print("[MCP] Connected to 3 category servers over stdio (oscar-erp / -inventory-wms / -logistics-tms).")

    async def _stop_transport(self) -> None:
        """Tear down the MCP client gateway (and its server subprocesses), if started."""
        if self._mcp_gateway is not None:
            await self._mcp_gateway.aclose()
            self._mcp_gateway = None
            self._tool_invoker = build_inproc_invoker()

    async def _run_loop(
        self, trace: List[Dict[str, Any]], max_loops: int, verbose: bool
    ) -> List[Dict[str, Any]]:
        """The closed ReAct loop body (transport is already established by `run`)."""
        while True:

            ledger = self.store.snapshot()

            # Circuit breaker: never exceed the hard loop bound.
            if ledger.metadata.loop_count > max_loops:
                self.store.mutate(
                    {"status": {"escalation_reason": "circuit_breaker_loop_limit"}}
                )
                if verbose:
                    print("[CIRCUIT BREAKER] loop_count exceeded; escalating to human.")
                break

            ledger_json = self.store.snapshot().model_dump_json(indent=2)
            scratchpad = self._render_scratchpad(trace)
            try:
                decision = await self._reason(ledger_json, scratchpad)
            except LLMUnavailableError as exc:
                # Reasoning core unreachable (e.g. quota exhausted). Escalate to HUMAN TAKEOVER
                # instead of crashing — the correct enterprise failure mode.
                LedgerStore.append_raw_log("orchestrator", f"LLM_UNAVAILABLE {exc}")
                self.store.mutate(
                    {"status": {"escalation_reason": "llm_quota_exhausted"}}
                )
                if verbose:
                    print(f"[HUMAN TAKEOVER] Reasoning core unavailable: {exc}")
                break
            thought = str(decision["thought"])
            action = cast(Dict[str, Any], decision["action"])
            tool = str(action.get("tool", "DONE"))
            args = cast(Dict[str, Any], action.get("args", {}))

            if verbose:
                print(f"\n--- Loop {ledger.metadata.loop_count} ---")
                print(f"Thought: {thought}")
                print(f"Action: {tool} {args}")

            LedgerStore.append_raw_log(
                "orchestrator",
                f"loop={ledger.metadata.loop_count} thought={thought!r} action={tool} args={args}",
            )

            # Narrate the reasoning + chosen action to the UI (observational).
            self._emit("thought", thought, loop=ledger.metadata.loop_count)
            self._emit("action", tool, tool=tool, args=args)

            if tool == "DONE":
                self.store.mutate({"status": {"goal_achieved": True}})
                trace.append({"thought": thought, "tool": tool, "args": args, "output": None})
                self._emit("resolution", "Incident assessment complete.")
                if verbose:
                    print("[DONE] Orchestrator reached terminal state.")
                break

            if tool not in KNOWN_TOOLS:
                # Unknown tool syntax -> record an observation and let the model recover.
                LedgerStore.append_raw_log("orchestrator", f"UNKNOWN_TOOL={tool}")
                trace.append(
                    {
                        "thought": thought,
                        "tool": tool,
                        "args": args,
                        "output": {"error": f"unknown tool '{tool}'"},
                    }
                )
                self.store.increment_loop()
                continue

            output = await self._dispatch_tool(tool, args)
            observation: Any = output["result"]

            # Narrate the tool's result as a data-rich plain-English line for the UI.
            self._emit("observation", self._humanize_observation(tool, observation), tool=tool)
            patch = self._mutation_for(tool, output)

            mutation_ok = True
            if patch:
                # Final safety net: a rejected mutation becomes a recoverable error Observation
                # instead of crashing the loop. The State Mutation Layer stays the authority.
                try:
                    self.store.mutate(patch)
                except ValidationError as exc:
                    LedgerStore.append_raw_log("orchestrator", f"MUTATION_REJECTED patch={patch} err={exc}")
                    observation = {"error": f"state mutation rejected: {exc.error_count()} invalid field(s)"}
                    mutation_ok = False

            if verbose:
                print(f"Observation: {json.dumps(observation)}")

            # UPDATE STATE LEDGER node: LoopCount++.
            self.store.increment_loop()
            # Append the full turn (incl. Observation) so the next cycle's scratchpad carries
            # this tool's result back into the model's context.
            trace.append(
                {"thought": thought, "tool": tool, "args": args, "output": observation}
            )

            # Committing a strategy is the resolving action: once a valid strategy is selected
            # the goal is achieved, so we close the loop here (no extra DONE turn).
            if tool == "commit_strategy" and mutation_ok:
                # ALT_SUPPLIER hands off to the async negotiation sub-graph, which counts as part
                # of THIS committed turn (no extra increment_loop) and may trigger the spend
                # guardrail / HITL.
                committed = str(args.get("strategy_type", ""))
                if committed == "ALT_SUPPLIER":
                    await self._maybe_negotiate(verbose)

                # Only mark RESOLVED if the guardrail did not end BREACHED. A human REJECTION
                # leaves guardrail_status=BREACHED, so the incident stays escalated.
                final = self.store.snapshot()
                if final.status.guardrail_status == "BREACHED":
                    self._emit(
                        "resolution",
                        self._summarize_resolution(final),
                        resolved=False,
                    )
                    if verbose:
                        print(
                            "[HUMAN TAKEOVER] Guardrail breached and not overridden; "
                            "incident escalated (not auto-resolved)."
                        )
                else:
                    self.store.mutate({"status": {"goal_achieved": True}})
                    # Re-snapshot so the summary reflects goal_achieved=True.
                    self._emit(
                        "resolution",
                        self._summarize_resolution(self.store.snapshot()),
                        resolved=True,
                    )
                    if verbose:
                        print("[GOAL ACHIEVED] Strategy committed; incident resolved.")
                break

        return trace


# Fixed incident resources (no scenario presets). The demo has a single realistic lever — the
# order quantity — and the agent's strategy emerges from comparing it against these finite
# resources: as the quantity grows past each resource the option closes, escalating
# INTERNAL_TRANSFER -> AIR_FREIGHT -> ALT_SUPPLIER. A second independent lever — the shipment
# delay (days) — drives the dynamic financial exposure via simulate_finance.
INTERNAL_TRANSFER_SURPLUS_UNITS = int(os.environ.get("INTERNAL_TRANSFER_SURPLUS_UNITS", "350"))
AIR_FREIGHT_CAPACITY_UNITS = int(os.environ.get("AIR_FREIGHT_CAPACITY_UNITS", "420"))
# Baseline enterprise revenue exposure (drives penalty + downtime math).
REVENUE_AT_RISK_USD = float(os.environ.get("REVENUE_AT_RISK_USD", "75000"))
# Observed shipment slip in days — the second live lever. Bigger => bigger loss.
DELAY_DAYS = int(os.environ.get("DELAY_DAYS", "9"))


async def main() -> None:
    """Demo entry point: initialize the target incident and run the async loop.

    Driven by two independent env-configurable levers: ORDER_QUANTITY (which mitigation is
    feasible, and thus the agent's choice + spend) and DELAY_DAYS (the shipment slip, which
    drives the dynamic projected loss).
    """
    STORE.init_incident(
        target_sku="SKU-99",
        primary_supplier_id="SUP-A",
        active_contract_id="CTR-4471",
        current_purchase_order_id="PO-88123",
        impacted_plants=["PLANT-2"],
        inventory_days_remaining=2,
        production_shutdown_hours=48,
        revenue_at_risk_usd=REVENUE_AT_RISK_USD,
        transferable_units=INTERNAL_TRANSFER_SURPLUS_UNITS,
        air_freight_available=True,
        air_freight_capacity_units=AIR_FREIGHT_CAPACITY_UNITS,
        # Seed the ledger-visible replacement quantity from the same value the commander uses
        # for spend, so the feasibility check and the guardrail's math read one number.
        replacement_order_qty=ORDER_QUANTITY,
        delay_days=DELAY_DAYS,
    )
    commander = IncidentCommander()

    mode = "LLM (Gemini)" if commander.llm_enabled else "OFFLINE deterministic planner"
    print(f"Incident Commander reasoning core: {mode} | model={GEMINI_MODEL}")
    print(
        f"Levers: order_qty={ORDER_QUANTITY}, delay_days={DELAY_DAYS} | "
        f"resources: surplus={INTERNAL_TRANSFER_SURPLUS_UNITS}, air_capacity={AIR_FREIGHT_CAPACITY_UNITS}"
    )
    await commander.run()
    print("\nFinal ledger:")
    print(STORE.snapshot().model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())


__all__ = ["IncidentCommander", "TOOL_REGISTRY", "GEMINI_MODEL", "SYSTEM_PROMPT"]
