"""
orchestrator.py
===============
The central *Incident Commander* agent (Section 2) — the LLM reasoning core.

This module implements the closed ReAct (Reason + Act) loop that drives the entire
system. Each cycle the Orchestrator:

    1. READS the serialized JSON of the Structured State Ledger from `ledger_store.STORE`.
    2. REASONS: the LLM emits an explicit `Thought:` line identifying data gaps / options.
    3. ACTS: the LLM selects its Next Best Action by emitting a tool call that must match
       one of our registered MCP Observation Tools or Decision Helpers.
    4. The tool is dispatched deterministically in Python; its parsed primitives are
       committed back to the ledger via the State Mutation Layer (`STORE.mutate`).
    5. `loop_count` is incremented; the circuit breaker forces a stop past 10 loops.

Reasoning core:
- Uses the unified Google Gen AI SDK: `from google import genai`.
- The API key is read securely from the environment (`GEMINI_API_KEY`) — never hardcoded.
- The model is swappable via `GEMINI_MODEL` (defaults to `gemini-2.5-pro`).

Resilience:
- If the SDK or API key is unavailable, the module still imports cleanly and falls back
  to a deterministic offline planner so the loop remains testable end-to-end without a
  live model. Set `GEMINI_API_KEY` to engage the real LLM reasoning core.
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
import mcp_server
import decision_helpers
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
# Hard circuit-breaker bound mirrored from the schema (loop_count le=11, escalate past 10).
MAX_LOOPS = 10


# Order quantity for the mitigation purchase — the realistic per-incident variable that
# drives total spend (unit price * qty) against the FIXED spend-authority limit. Configurable
# via env so a demo can flip between the auto-approve path (low qty) and the HITL path (high
# qty) without code edits. Default 500 => 500 * ~$44 = ~$22k, which exceeds the $20k limit.
ORDER_QUANTITY = int(os.environ.get("ORDER_QUANTITY", "500"))

# HITL decision channel: 'cli' prompts the operator y/n at the terminal; 'auto_approve' /
# 'auto_reject' are non-interactive defaults for tests/CI. The Streamlit dashboard injects
# its own callable (Approve/Reject buttons) directly, bypassing this env switch.
HITL_MODE = os.environ.get("HITL_MODE", "cli")

# DEMO TOGGLE (default OFF): force the offline planner to COMMIT the ALT_SUPPLIER strategy
# regardless of the recorded strategy_scores. This does NOT alter the scores themselves
# (they stay honest at ALT_SUPPLIER 60.35 / INTERNAL_TRANSFER 80.0 / AIR_FREIGHT 65.28); it
# only overrides the terminal SELECTION so the ALT_SUPPLIER path runs — which is the ONLY
# path that reaches the negotiation sub-graph and therefore the Financial Spend-Authority
# Guardrail + HITL. Needed because the honest optimal pick is INTERNAL_TRANSFER, which never
# spends and so never exercises the guardrail. Off by default → normal runs are unchanged.
FORCE_ALT_SUPPLIER = os.environ.get("FORCE_ALT_SUPPLIER", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


# A human-in-the-loop decision provider: given the breach details, return True to APPROVE
# the over-limit spend or False to REJECT it. It may be SYNC (returns bool) OR ASYNC
# (returns an awaitable bool) — `_enforce_spend_guardrail` awaits either transparently, so
# the Streamlit dashboard can inject a plain sync callback while the CLI stays non-blocking.
HumanDecisionFn = Callable[[SpendAuthorityResult], Union[bool, Awaitable[bool]]]


async def _default_human_decision(result: SpendAuthorityResult) -> bool:
    """Default HITL provider driven by HITL_MODE (cli | auto_approve | auto_reject).

    Returns True to APPROVE the over-limit purchase, False to REJECT it. The auto_* modes
    make deterministic choices for offline tests/CI. The CLI mode reads operator input via
    `asyncio.to_thread(input, ...)` so the blocking terminal prompt runs on a worker thread
    and NEVER stalls the asyncio event loop (keeps concurrent tasks / UI telemetry alive).
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
        # Offload blocking input() to a worker thread — does NOT block the event loop.
        answer = (await asyncio.to_thread(input, prompt)).strip().lower()
    except EOFError:  # non-interactive stdin -> safe default is to REJECT
        return False
    return answer in ("y", "yes")



# ---------------------------------------------------------------------------- #
# Tool registry — the ONLY actions the Orchestrator may select each turn.

# Maps the tool name the LLM emits to a concrete Python callable.
# ---------------------------------------------------------------------------- #
ToolFn = Callable[..., Any]

TOOL_REGISTRY: Dict[str, ToolFn] = {
    # MCP Observation Tools (I/O bound).
    "query_erp": mcp_server.query_erp,
    "query_inventory": mcp_server.query_inventory,
    "query_shipment_tracking": mcp_server.query_shipment_tracking,
    "extract_contract_rules": mcp_server.extract_contract_rules,
    # Decision Helpers (deterministic math).
    "simulate_finance": decision_helpers.simulate_finance,
    "score_strategy": decision_helpers.score_strategy,
    "policy_check": decision_helpers.policy_check,
}

# Allowed mitigation strategies — MUST match the Literal set in schema.MitigationState.
# Published to the LLM so it never invents an out-of-schema value (e.g. "EXPEDITE").
VALID_STRATEGIES = ("ALT_SUPPLIER", "INTERNAL_TRANSFER", "AIR_FREIGHT")

# Virtual (non-registry) actions the orchestrator handles itself, in addition to the
# concrete TOOL_REGISTRY functions. `commit_strategy` is a selection decision (not a
# computational helper); `DONE` is the terminal signal.
CONTROL_ACTIONS = ("commit_strategy", "DONE")
KNOWN_TOOLS = frozenset(TOOL_REGISTRY) | frozenset(CONTROL_ACTIONS)


# Human-readable tool contracts injected into the system prompt so the LLM emits valid
# tool-calling syntax. Kept terse and machine-oriented (no markdown prose in state).
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
{TOOL_CATALOG}
ONE-SHOT FORMAT EXAMPLE (this exact two-line structure is REQUIRED every turn):
Thought: The contract penalty rate is still 0.0, so I must parse the contract before I can simulate finance accurately.
Action: {{"tool": "extract_contract_rules", "args": {{"contract_id": "CTR-4471"}}}}

Respond with ONLY the 'Thought:' line and the 'Action:' line. No extra commentary.
"""


# ---------------------------------------------------------------------------- #
# Gemini client bootstrap (secure, resilient).
# ---------------------------------------------------------------------------- #
def _init_genai_client() -> Optional[Any]:
    """
    Build the unified Google Gen AI client from GEMINI_API_KEY. Returns None if the SDK is
    not installed or no key is present, allowing a deterministic offline fallback.
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
    """
    Central orchestrator. Wired directly to the `STORE` singleton for reads/mutations.
    """

    def __init__(
        self,
        store: Optional[LedgerStore] = None,
        *,
        order_quantity: int = ORDER_QUANTITY,
        spend_authority_limit_usd: float = SPEND_AUTHORITY_LIMIT_USD,
        human_decision: Optional[HumanDecisionFn] = None,
        force_alt_supplier: bool = FORCE_ALT_SUPPLIER,
    ) -> None:
        self.store = store or STORE
        self._client = _init_genai_client()
        # True when a live LLM core is available; else we use the offline planner.
        self.llm_enabled = self._client is not None
        # Order quantity is the realistic per-incident variable driving total spend against
        # the FIXED delegated authority limit — flip it to demo auto-approve vs HITL paths.
        self.order_quantity = order_quantity
        self.spend_authority_limit_usd = spend_authority_limit_usd
        # DEMO TOGGLE (default OFF via env): when True the offline planner commits
        # ALT_SUPPLIER regardless of scores so the spend-authority guardrail + HITL run.
        # Scores are NOT mutated — only the terminal selection is overridden. Exposed as a
        # constructor param (env-defaulted) so the CLI (--force-alt-supplier) and dashboard
        # (checkbox) can flip it per-run without touching os.environ.
        self.force_alt_supplier = force_alt_supplier

        # HITL decision provider (approve/reject an over-limit spend). Defaults to the
        # env-driven CLI/auto provider; the Streamlit dashboard injects its own callable.
        self.human_decision: HumanDecisionFn = human_decision or _default_human_decision


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
        Observation turns) so the model has memory of what it already did and observed —
        this is the Observation step that prevents blind, repeated tool calls.
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
        """Call Gemini via the SHARED resilient helper (`llm_utils.generate_with_retry`).

        The retry/backoff policy for 429/5xx (and daily-quota escalation) lives in one place
        so the Incident Commander and the Supplier-Persona sub-graph behave identically.
        """
        return await generate_with_retry(
            self._client,
            model=GEMINI_MODEL,
            contents=contents,
            config=config,
            source="orchestrator",
            log=LedgerStore.append_raw_log,
        )

    def _reason_offline(self, ledger_json: str) -> Dict[str, Any]:

        """
        Deterministic fallback planner used when no live model is configured.

        It walks the same information-gathering order a well-behaved LLM would follow,
        driven purely by which ledger fields are still unpopulated. This keeps the closed
        loop fully exercisable in CI / smoke tests without an API key.
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
        # 2) Quantify financial exposure now that the rate is known.
        if metrics.get("projected_total_loss_usd", 0.0) == 0.0:
            return {
                "thought": "Penalty rate known; simulate the financial impact of the delay.",
                "action": {"tool": "simulate_finance", "args": {"delay_days": 9}},
            }
        # 3) Evaluate (score) the candidate strategies — recording only, not selecting.
        for candidate in VALID_STRATEGIES:
            if candidate not in scores:
                return {
                    "thought": f"Evaluate {candidate} so I can compare options before committing.",
                    "action": {"tool": "score_strategy", "args": {"strategy_type": candidate}},
                }
        # 4) All scored: deliberately COMMIT the winning strategy.
        # Committing is the resolving action (the run loop treats it as terminal), so this
        # is the planner's final decision for this incident type — no separate DONE turn is
        # emitted here (that would be unreachable and waste a loop against the breaker bound).
        # DEMO OVERRIDE: when force_alt_supplier is set, commit ALT_SUPPLIER regardless of
        # the (untouched) scores so the negotiation + spend-authority guardrail + HITL run;
        # otherwise pick the honest highest-composite-score strategy (INTERNAL_TRANSFER here).
        if self.force_alt_supplier:
            thought = (
                f"Scores compared ({scores}); demo override forces ALT_SUPPLIER to exercise "
                "the spend-authority guardrail — commit it."
            )
            return {
                "thought": thought,
                "action": {"tool": "commit_strategy", "args": {"strategy_type": "ALT_SUPPLIER"}},
            }
        best = max(scores, key=lambda k: scores[k])
        return {
            "thought": f"Scores compared ({scores}); {best} is best — commit it.",
            "action": {"tool": "commit_strategy", "args": {"strategy_type": best}},
        }


    @staticmethod
    def _extract_json_object(blob: str) -> Optional[Dict[str, Any]]:
        """Best-effort extraction of a single JSON object from messy LLM text.

        Live models often wrap the Action JSON in markdown code fences (```json ... ```),
        add stray prose, or pretty-print across multiple lines. A naive greedy `\\{.*\\}`
        can swallow trailing junk and fail to parse. This helper:
          1. strips ```json / ``` fences,
          2. finds the FIRST '{' then scans with brace-depth counting to the MATCHING '}'
             (so it isolates exactly one balanced object, ignoring anything after it),
          3. attempts json.loads on that balanced substring.
        Returns the parsed dict, or None if no valid object is found.
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

        Thinking models (e.g. gemini-2.5-flash) often emit multi-line reasoning after the
        'Thought:' token (captured via DOTALL) and frequently wrap the Action JSON in
        markdown code fences. We isolate the Action payload after the 'Action:' token and
        parse it with a brace-matching, fence-tolerant extractor so valid tool calls are
        NOT silently downgraded to DONE (which would halt the agent prematurely).
        """
        # Capture multi-line thought up to the Action token (or end of text).
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\n\s*Action:|\Z)", text, re.DOTALL | re.IGNORECASE
        )
        thought = thought_match.group(1).strip() if thought_match else ""

        # Grab everything AFTER the first 'Action:' token, then extract one JSON object
        # from it (tolerating code fences / pretty-printing / trailing prose).
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
    def _dispatch_tool(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered tool, injecting the ledger snapshot where required.

        Performs defensive argument validation BEFORE running the tool so a semantically
        reasonable but schema-noncompliant LLM output (e.g. strategy_type="EXPEDITE")
        becomes a recoverable error Observation instead of a downstream crash.
        """
        # SECURITY (§5.1): sanitize EVERY dispatched tool's args (defense-in-depth), not
        # just write actions. Any LLM-supplied string is scanned for prompt-injection /
        # command-escape patterns + length bounds BEFORE the tool runs; a hit aborts with a
        # recoverable error Observation (the loop continues safely).
        try:
            sanitize_write_payload(args)
        except InjectionAttemptError as exc:
            LedgerStore.append_raw_log("security", f"INJECTION_BLOCKED tool={tool} args={args} err={exc}")
            return {"result": {"error": f"tool call blocked by security layer: {exc}"}}

        # Guard: reject out-of-schema strategy names up front and tell the model why.
        # Applies to BOTH evaluation (score_strategy) and selection (commit_strategy).
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

        # commit_strategy is a selection action (not a computational helper): it simply
        # confirms the chosen strategy, which _mutation_for then writes to active_strategy.
        # (Its args were ALREADY sanitized by the single entry-point sweep above — no
        # duplicate scan needed here.)
        if tool == "commit_strategy":
            return {"result": {"committed_strategy": str(args.get("strategy_type", ""))}}



        fn = TOOL_REGISTRY[tool]
        # Decision Helpers that reason over full state receive the live snapshot.
        if tool in ("simulate_finance", "score_strategy"):
            args = {**args, "state_ledger_snapshot": self.store.snapshot_dict()}
        # RESILIENCE: a live LLM can emit missing/extra/mistyped args for any tool. Guard the
        # call so a bad signature (e.g. omitting a required arg) becomes a recoverable error
        # Observation instead of crashing the ReAct loop — same philosophy as the mutation
        # ValidationError net and the strategy/injection guards above.
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
        # After the isinstance guard, treat the payload as a typed str-keyed mapping.
        result = cast(Dict[str, Any], raw)

        if tool == "extract_contract_rules" and result.get("found"):
            return {"context": {"contracted_penalty_rate": result["contracted_penalty_rate"]}}
        if tool == "query_shipment_tracking" and result.get("found"):
            # Record ONLY the observed delay as a dedicated `delay_days` primitive.
            # IMPORTANT (fix): do NOT overwrite `production_shutdown_hours` here — a shipment
            # delay is NOT the same as shutdown hours; the on-hand inventory buffer absorbs
            # part of it. `simulate_finance` remains the SOLE authority on shutdown/downtime
            # math (it derives shutdown days from delay beyond inventory_days_remaining).
            delay_days = int(result.get("delay_days", 0))
            if delay_days > 0:
                return {"metrics": {"delay_days": delay_days}}
            return {}

        if tool == "simulate_finance":
            # Persist the computed financial primitives to the ledger so the single source
            # of truth captures the incident's financial exposure (not just static revenue).
            return {
                "metrics": {
                    "daily_penalty_usd": result["daily_penalty_usd"],
                    "projected_total_loss_usd": result["projected_total_loss_usd"],
                }
            }
        if tool == "score_strategy":
            # EVALUATE only: record the score under strategy_scores WITHOUT selecting it.
            # A leaf patch is sufficient — the State Mutation Layer's deep-merge preserves
            # previously-scored sibling strategies, so we needn't rebuild the dict here.
            strat = str(result["strategy_type"])
            return {"mitigation": {"strategy_scores": {strat: result["composite_score"]}}}
        if tool == "commit_strategy":
            # SELECT: a deliberate, separate decision that sets the active strategy after
            # the agent has compared the recorded strategy_scores.
            strat = str(result.get("committed_strategy", ""))
            if strat in VALID_STRATEGIES:
                return {"mitigation": {"active_strategy": strat}}
            return {}
        # Observation tools that only inform reasoning need no direct mutation this turn.
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
            # The Observation is the actual tool output fed back to the model (the key
            # missing piece that turns blind repetition into grounded reasoning).
            lines.append(f"  Observation: {json.dumps(turn.get('output'))}")
        return "\n".join(lines)

    async def _maybe_negotiate(self, verbose: bool) -> None:
        """
        After an ALT_SUPPLIER commit, negotiate with ALL alternate suppliers CONCURRENTLY
        (asyncio.gather) and select the lowest-price winner.

        CRITICAL BUDGET RULE: the entire multi-vendor negotiation (all internal LLM-to-LLM
        volleys across suppliers) is ONE orchestrator action. This method `await`s the
        sub-graphs but does NOT call `increment_loop()` — the surrounding run() turn already
        counts as the single loop. Each sub-graph keeps its own isolated turn counter.

        CONCURRENCY LOCK: the orchestrator holds the master status lock. It writes
        `negotiation_status=IN_PROGRESS` once here, runs the suppliers with
        `write_status=False` (so they never race on the shared field), then writes the
        single final SUCCESS/FAILED after evaluating results.
        """
        snapshot = self.store.snapshot()
        # Discover alternate vendors THROUGH the MCP layer (never a direct DB import): the
        # orchestrator stays decoupled from the ERP data source, exactly like production.
        alts = cast(
            List[Dict[str, Any]],
            mcp_server.query_alternate_suppliers(snapshot.context.target_sku),
        )
        incident_id = str(snapshot.metadata.id)

        # Buyer's willingness-to-pay ceiling is derived from OUR OWN economics, not the
        # vendor's price: it's the primary supplier's unit cost plus an acceptable premium
        # we'll tolerate to avoid the far larger downtime loss. (Sourced from ERP via MCP.)
        primary = cast(Dict[str, Any], mcp_server.query_erp(snapshot.context.target_sku))
        primary_unit_cost = float(primary.get("unit_cost_usd", 42.5)) if primary.get("found") else 42.5
        ACCEPTABLE_PREMIUM = 1.10  # willing to pay up to 10% over primary cost to resolve.
        buyer_ceiling = round(primary_unit_cost * ACCEPTABLE_PREMIUM, 2)

        if not alts:
            # No alternate vendors to negotiate with -> clean FAILED, no crash.
            self.store.mutate({"mitigation": {"negotiation_status": "FAILED"}})
            if verbose:
                print("[SUB-GRAPH] No alternate suppliers available; negotiation FAILED.")
            return

        # Orchestrator takes the master status lock BEFORE spawning concurrent negotiations.
        self.store.mutate({"mitigation": {"negotiation_status": "IN_PROGRESS"}})
        if verbose:
            print(f"[SUB-GRAPH] Negotiating {len(alts)} suppliers concurrently (asyncio.gather) ...")

        # Build one negotiation coroutine per alternate supplier with a defensible economic
        # model:
        #   - buyer ceiling  = OUR willingness-to-pay (primary cost + tolerated premium),
        #                      shared across vendors — it's about our economics, not theirs.
        #   - supplier floor = that vendor's TRUE unit cost (a rational vendor won't sell
        #                      below cost); distinct per vendor, so quotes genuinely differ.
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

        # CONCURRENCY RESILIENCE (fix): pass return_exceptions=True so a single vendor
        # sub-graph failure (timeout / dropped connection / LLM error) does NOT abort the
        # whole gather and crash the orchestrator. Failures come back as Exception objects
        # in the results list; we log and drop them, then proceed with whatever succeeded.
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


        # EMPTY-SEQUENCE GUARD: never call min() on an empty list.
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

        # Winning supplier id is logged for the audit trail AND persisted (now in-schema).
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
        if verbose:
            print(f"[SUB-GRAPH] Best quote: {best}")

        # --- FINANCIAL SPEND-AUTHORITY GUARDRAIL (§5.2) --------------------------------- #
        # A deal is on the table. Before it is treated as resolved, the deterministic
        # guardrail checks whether the total spend (unit price * order quantity) is within
        # the agent's delegated authority. Over-limit spend hard-forks to a HUMAN-IN-THE-LOOP
        # approve/reject decision — the LLM has NO say in this barrier.
        await self._enforce_spend_guardrail(best, verbose)

    async def _enforce_spend_guardrail(self, best: TermsDict, verbose: bool) -> None:

        """Run the spend-authority guardrail on a won deal and drive the HITL decision (§5.2).

        Outcomes written to the ledger:
        - within authority        -> guardrail PASSED, deal stands (resolved by caller).
        - over authority, APPROVED -> human override: guardrail PASSED, deal stands.
        - over authority, REJECTED -> guardrail BREACHED, active_strategy cleared (no PO),
          negotiation_status FAILED, escalation_reason set — incident returned to human.
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
            # Spend is within delegated authority -> auto-approved, no human needed.
            self.store.mutate({"status": {"guardrail_status": "PASSED"}})
            if verbose:
                print(f"[GUARDRAIL] {check.reason}")
            return

        # Over authority -> pause and ask a human (HITL). Record the breach first so any
        # watcher (dashboard) can render the pending decision, then obtain the verdict.
        self.store.mutate(
            {"status": {"guardrail_status": "BREACHED", "escalation_reason": check.reason}}
        )
        if verbose:
            print(f"[GUARDRAIL] BREACHED — {check.reason}")

        # Obtain the human verdict. The provider may be SYNC (returns bool) or ASYNC
        # (returns an awaitable) — await it transparently so the CLI prompt runs off the
        # event loop while a dashboard can still inject a plain sync callback.
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
            if verbose:
                print(f"[HITL] APPROVED — PO authorized with {supplier_id} for ${check.spend_usd:,.2f}.")
            return

        # Human rejected: cancel the purchase (no PO), keep BREACHED, escalate for manual
        # handling. Clearing active_strategy signals no mitigation was executed.
        self.store.mutate(
            {
                "mitigation": {"active_strategy": "NONE", "negotiation_status": "FAILED"},
                "status": {
                    "guardrail_status": "BREACHED",
                    "escalation_reason": "human_rejected_over_limit_spend",
                },
            }
        )
        if verbose:
            print(f"[HITL] REJECTED — no PO placed; incident escalated for manual handling.")


    async def run(self, max_loops: int = MAX_LOOPS, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Execute the autonomous reasoning loop until DONE, goal achieved, or the circuit
        breaker trips (loop_count > 10). Returns the ordered trace of turns for inspection.

        A running ReAct scratchpad (Thought / Action / Observation per turn) is maintained
        and injected into each reasoning call, giving the model memory of prior tool
        observations so it converges instead of repeating the same action.

        This is an async coroutine so it can `await` the negotiation sub-graph, genuinely
        suspending and resuming per the §4.3 handoff contract.
        """
        trace: List[Dict[str, Any]] = []

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
                # Reasoning core unreachable (e.g. quota exhausted). Escalate to the
                # HUMAN TAKEOVER terminal node instead of crashing — the correct
                # enterprise failure mode when the agent cannot think.
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

            if tool == "DONE":
                self.store.mutate({"status": {"goal_achieved": True}})
                trace.append({"thought": thought, "tool": tool, "args": args, "output": None})
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

            output = self._dispatch_tool(tool, args)
            observation: Any = output["result"]
            patch = self._mutation_for(tool, output)
            mutation_ok = True
            if patch:
                # Final safety net: a rejected mutation (bad primitive the guards missed)
                # is converted into a recoverable error Observation instead of crashing
                # the whole loop. The State Mutation Layer stays the authority on validity.
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
            # Append the full turn (incl. Observation) so the next cycle's scratchpad
            # carries this tool's result back into the model's context.
            trace.append(
                {"thought": thought, "tool": tool, "args": args, "output": observation}
            )

            # Committing a strategy is the resolving action for this incident type: once a
            # valid strategy is selected the goal is achieved, so we close the loop here
            # (rather than requiring an extra DONE turn that would risk the circuit breaker).
            if tool == "commit_strategy" and mutation_ok:
                # If ALT_SUPPLIER was chosen, hand off to the async negotiation sub-graph
                # to secure terms with the alternate vendor. The whole sub-graph counts as
                # part of THIS single committed turn — no extra increment_loop() is called.
                # The negotiation may trigger the Financial Spend-Authority Guardrail,
                # which can hard-fork to a HUMAN-IN-THE-LOOP approve/reject decision.
                committed = str(args.get("strategy_type", ""))
                if committed == "ALT_SUPPLIER":
                    await self._maybe_negotiate(verbose)

                # Only mark RESOLVED if the guardrail did not end BREACHED (i.e. no spend
                # limit hit, or a human APPROVED the override). A human REJECTION leaves
                # guardrail_status=BREACHED, so the incident stays escalated for manual
                # handling rather than being falsely reported as resolved.
                final = self.store.snapshot()
                if final.status.guardrail_status == "BREACHED":
                    if verbose:
                        print(
                            "[HUMAN TAKEOVER] Guardrail breached and not overridden; "
                            "incident escalated (not auto-resolved)."
                        )
                else:
                    self.store.mutate({"status": {"goal_achieved": True}})
                    if verbose:
                        print("[GOAL ACHIEVED] Strategy committed; incident resolved.")
                break


        return trace


async def main() -> None:
    """Demo entry point: initialize the target incident and run the async loop."""
    STORE.init_incident(
        target_sku="SKU-99",
        primary_supplier_id="SUP-A",
        active_contract_id="CTR-4471",
        current_purchase_order_id="PO-88123",
        impacted_plants=["PLANT-2"],
        inventory_days_remaining=2,
        production_shutdown_hours=48,
        revenue_at_risk_usd=4200.0,
    )
    commander = IncidentCommander()
    mode = "LLM (Gemini)" if commander.llm_enabled else "OFFLINE deterministic planner"
    print(f"Incident Commander reasoning core: {mode} | model={GEMINI_MODEL}")
    await commander.run()
    print("\nFinal ledger:")
    print(STORE.snapshot().model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())


__all__ = ["IncidentCommander", "TOOL_REGISTRY", "GEMINI_MODEL", "SYSTEM_PROMPT"]
