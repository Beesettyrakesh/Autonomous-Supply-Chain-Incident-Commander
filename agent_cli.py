"""
Terminal operator console for the Autonomous Supply Chain Incident Commander.

Drives the same `IncidentCommander` async ReAct loop as the dashboard, narrating each step
(Reasoning / Action / Finding / Negotiation / Human Review / Resolution) as a colorized feed
and injecting a human-in-the-loop decision provider for the spend-authority guardrail. It is
purely a presentation + operator-input surface — it never changes the agent's control flow.
Runs the deterministic offline planner by default; pass `--live` for the Gemini core. See
`--help` for all options.
"""


from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any, Dict

# --------------------------------------------------------------------------- #
# Optional colorama — degrade gracefully to plain text if it isn't installed.
# --------------------------------------------------------------------------- #
_colorama_available = True
try:
    from colorama import Fore, Style
    from colorama import init as _colorama_init
except ImportError:  # pragma: no cover - colorama is an optional presentation dep
    _colorama_available = False

    # Bind harmless fallbacks so references below are always defined (no-color path).
    Fore = Style = None  # type: ignore[assignment]

    def _colorama_init(*_args: Any, **_kwargs: Any) -> None:  # type: ignore[misc]
        return None


class _NoColor:
    """Stand-in for colorama's Fore/Style whose attributes are all empty strings.

    Lets the renderer reference e.g. `C.CYAN` unconditionally; when color is disabled the
    attribute resolves to "" so the same f-strings produce clean, unstyled output.
    """

    def __getattr__(self, _name: str) -> str:
        return ""


# Business-friendly purpose per tool — mirrors dashboard._TOOL_PURPOSE so the CLI and cockpit
# tell the same story. Kept local so this module has no Streamlit import.
_TOOL_PURPOSE: Dict[str, str] = {
    "extract_contract_rules": "Parsing the supplier contract for the late-delivery penalty",
    "query_shipment_tracking": "Retrieving the latest shipment status and delay",
    "query_erp": "Looking up the ERP purchase-order and vendor master",
    "query_inventory": "Checking plant inventory cover and transfer options",
    "simulate_finance": "Quantifying the financial exposure of the delay",
    "score_strategy": "Scoring a candidate mitigation strategy",
    "commit_strategy": "Committing the chosen mitigation strategy",
    "policy_check": "Verifying vendor and spend against purchasing policy",
    "DONE": "Concluding the incident assessment",
}


class CliRenderer:
    """Translate the orchestrator's `on_event` stream into a colorized terminal feed.

    Registered as the commander's `on_event` sink. Each event is a
    (kind, message, data) triple mapped to a colored label (or a chat bubble for
    negotiation). Purely observational — it never touches control flow.
    """

    def __init__(self, color: bool) -> None:
        self._color = color
        self.C: Any = Fore if color and _colorama_available else _NoColor()
        self.S: Any = Style if color and _colorama_available else _NoColor()

    def _label(self, text: str, color: str) -> str:
        return f"{color}{self.S.BRIGHT}{text}{self.S.RESET_ALL}"

    def __call__(self, kind: str, message: str, data: Dict[str, Any]) -> None:
        C, S = self.C, self.S
        if kind == "thought":
            print(f"\n{self._label('  Reasoning', C.CYAN)} — {message}")
        elif kind == "action":
            tool = str(data.get("tool") or message).strip()
            purpose = _TOOL_PURPOSE.get(tool)
            body = f"{purpose} — calling `{tool}`" if purpose else f"calling `{tool}`"
            print(f"{self._label('  Action', C.YELLOW)} — {body}")
        elif kind == "observation":
            body = str(message).strip()
            if body:
                print(f"{self._label('  Finding', C.WHITE)} — {body}")
        elif kind == "negotiation":
            self._render_negotiation(message, data)
        elif kind == "guardrail":
            passed = bool(data.get("passed"))
            color = C.GREEN if passed else C.RED
            print(f"\n{color}{S.BRIGHT}{message}{S.RESET_ALL}")
        elif kind == "hitl":
            print(f"{self._label('  Human Review', C.YELLOW)} — {message}")
        elif kind == "resolution":
            # The "Incident assessment complete." heartbeat is internal — skip the noise.
            if message == "Incident assessment complete.":
                return
            resolved = bool(data.get("resolved", True))
            color = C.GREEN if resolved else C.RED
            print(f"\n{color}{S.BRIGHT}{message}{S.RESET_ALL}")

    def _render_negotiation(self, message: str, data: Dict[str, Any]) -> None:
        """Render a negotiation line as a role-tagged chat bubble (buyer / supplier / system)."""
        C, S = self.C, self.S
        role = str(data.get("role", "system"))
        if role == "buyer":
            print(f"    {C.BLUE}🧑‍💼 Buyer{S.RESET_ALL}: {message}")
        elif role == "supplier":
            sid = str(data.get("supplier", "Supplier"))
            print(f"    {C.MAGENTA}🏭 {sid}{S.RESET_ALL}: {message}")
        else:
            print(f"  {C.MAGENTA}{message}{S.RESET_ALL}")


def _build_human_decision(mode: str, renderer: CliRenderer):
    """Return a sync human-decision provider for the spend-authority guardrail (HITL).

    'approve'/'reject' are non-interactive verdicts for scripts/CI; 'prompt' reads an
    interactive y/N at the terminal. The orchestrator awaits this callable transparently, so a
    plain sync function is fine.
    """
    C, S = renderer.C, renderer.S

    def _decide(result: Any) -> bool:
        if mode == "approve":
            return True
        if mode == "reject":
            return False
        # Interactive prompt.
        banner = (
            f"\n{C.YELLOW}{S.BRIGHT}{'=' * 68}\n"
            "[HUMAN-IN-THE-LOOP] Spend exceeds the agent's delegated authority.\n"
            f"  Supplier : {result.supplier_id}\n"
            f"  Spend    : ${result.spend_usd:,.2f}\n"
            f"  Limit    : ${result.limit_usd:,.2f}\n"
            f"  Reason   : {result.reason}\n"
            f"{'=' * 68}{S.RESET_ALL}\n"
            "Approve this over-limit purchase? [y/N]: "
        )
        try:
            answer = input(banner).strip().lower()
        except EOFError:  # non-interactive stdin -> safe default is REJECT
            return False
        return answer in ("y", "yes")

    return _decide


def _parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agent_cli.py",
        description=(
            "Terminal console for the Autonomous Supply Chain Incident Commander — drives "
            "the same autonomous ReAct loop as the dashboard, with colorized step narration "
            "and a human-in-the-loop spend-authority gate."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--qty",
        type=int,
        default=500,
        help="Order quantity — the SINGLE lever the agent's strategy emerges from. It is "
        "compared against the internal transfer surplus (350) and air cargo capacity (420): "
        "<=350 -> INTERNAL_TRANSFER; 351-420 -> AIR_FREIGHT; >420 -> ALT_SUPPLIER (negotiate). "
        "It also sets total spend (unit price x qty) vs the $20k authority. (default: 500)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=9,
        help="Observed shipment delay in days — the second, independent lever. It drives the "
        "DYNAMIC projected loss (bigger delay => bigger exposure), not the strategy choice. "
        "(default: 9)",
    )
    parser.add_argument(
        "--surplus",
        type=int,
        default=350,
        help="Internal transfer surplus (units PLANT-1 can spare). INTERNAL_TRANSFER is "
        "feasible only when the order quantity is within this. (default: 350)",
    )
    parser.add_argument(
        "--air-capacity",
        dest="air_capacity",
        type=int,
        default=420,
        help="Finite air cargo capacity (units). AIR_FREIGHT is feasible only when the order "
        "quantity is within this. (default: 420)",
    )

    parser.add_argument(
        "--hitl",
        choices=["prompt", "approve", "reject"],
        default="prompt",
        help="How to resolve an over-limit spend breach. prompt = interactive y/N (default); "
        "approve/reject = non-interactive verdict for scripts/CI.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Engage the live Gemini reasoning core (uses your API quota). Default is the "
        "zero-cost deterministic offline planner.",
    )
    parser.add_argument(
        "--spend-limit",
        type=float,
        default=None,
        help="Override the delegated spend-authority limit in USD (default: the locked "
        "SPEND_AUTHORITY_LIMIT_USD, normally $20,000).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output (useful when piping to a file or a CI log).",
    )
    parser.add_argument(
        "--json",
        dest="dump_json",
        action="store_true",
        help="Print the final State Ledger as JSON at the end (audit / architecture proof).",
    )
    return parser.parse_args(argv)


def _configure_reasoning_core(live: bool) -> None:
    """Set the in-process env the commander reads at construction.

    Offline (default) forces the deterministic planner + scripted vendor so no LLM calls are
    made. --live leaves GEMINI_API_KEY as provided in the environment/.env and switches the
    vendor persona to the live LLM.
    """
    if live:
        os.environ["VENDOR_MODE"] = "llm"
        # GEMINI_API_KEY is not touched here — it must come from the environment / .env. If
        # absent, the orchestrator falls back to the offline planner automatically.
    else:
        os.environ["GEMINI_API_KEY"] = ""            # -> deterministic offline planner
        os.environ["VENDOR_MODE"] = "deterministic"  # -> scripted vendor (no LLM calls)


async def _run(args: argparse.Namespace) -> int:
    # Import AFTER env is configured so the modules read the intended VENDOR_MODE etc.
    from ledger_store import STORE
    from orchestrator import IncidentCommander, GEMINI_MODEL, REVENUE_AT_RISK_USD
    from guardrails import SPEND_AUTHORITY_LIMIT_USD

    color = not args.no_color
    if color and _colorama_available:
        _colorama_init(autoreset=False)

    renderer = CliRenderer(color=color)
    C, S = renderer.C, renderer.S

    spend_limit = args.spend_limit if args.spend_limit is not None else float(SPEND_AUTHORITY_LIMIT_USD)

    # Establish the single source of truth. The agent's strategy emerges from the order
    # quantity vs the finite resources below; the delay drives the dynamic projected loss.
    STORE.init_incident(
        target_sku="SKU-99",
        primary_supplier_id="SUP-A",
        active_contract_id="CTR-4471",
        current_purchase_order_id="PO-88123",
        impacted_plants=["PLANT-2"],
        inventory_days_remaining=2,
        production_shutdown_hours=48,
        revenue_at_risk_usd=REVENUE_AT_RISK_USD,
        transferable_units=args.surplus,
        air_freight_available=True,
        air_freight_capacity_units=args.air_capacity,
        replacement_order_qty=args.qty,
        delay_days=args.delay,
    )

    commander = IncidentCommander(
        order_quantity=args.qty,
        spend_authority_limit_usd=spend_limit,
        human_decision=_build_human_decision(args.hitl, renderer),
        on_event=renderer,
    )

    core = f"LIVE · {GEMINI_MODEL}" if commander.llm_enabled else "OFFLINE · deterministic planner"

    # ------------------------------ Banner ---------------------------------- #
    print(f"{C.CYAN}{S.BRIGHT}{'=' * 68}{S.RESET_ALL}")
    print(f"{C.CYAN}{S.BRIGHT}  AUTONOMOUS SUPPLY CHAIN INCIDENT COMMANDER — CLI{S.RESET_ALL}")
    print(f"{C.CYAN}{'=' * 68}{S.RESET_ALL}")
    print(f"  Incident  : SKU-99 shipment delayed {args.delay} days (SUP-A -> PLANT-2)")
    print(f"  Order qty : {args.qty}  |  Spend authority: ${spend_limit:,.0f}")
    print(f"  Resources : internal surplus {args.surplus} · air capacity {args.air_capacity} · alt-supplier pool")
    print(f"  HITL mode : {args.hitl}")
    print(f"  Core      : {core}")
    print(f"{C.CYAN}{'=' * 68}{S.RESET_ALL}")


    if args.live and not commander.llm_enabled:
        print(
            f"{C.YELLOW}Note: --live requested but no GEMINI_API_KEY was found; "
            f"falling back to the deterministic offline planner.{S.RESET_ALL}"
        )

    # ------------------------------ Run ------------------------------------- #
    await commander.run(verbose=False)

    # ------------------------------ Ledger dump (optional) ------------------ #
    if args.dump_json:
        print(f"\n{C.CYAN}{S.BRIGHT}Final State Ledger (single source of truth):{S.RESET_ALL}")
        print(STORE.snapshot().model_dump_json(indent=2))

    # Exit code reflects the outcome: 0 = resolved (or human-approved), 1 = escalated to a
    # human (guardrail breached and not overridden).
    final = STORE.snapshot()
    return 1 if final.status.guardrail_status == "BREACHED" else 0


def main(argv: Any = None) -> int:
    args = _parse_args(argv)
    _configure_reasoning_core(args.live)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
