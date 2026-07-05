"""
negotiation_agent.py
====================
Asynchronous Sub-Graph (Section 4.3) — the *Supplier Negotiation Agent*.

This is a genuine multi-agent sub-graph: our **Negotiation Agent** (an LLM) bargains with
an independent **Supplier-Persona** (a second LLM) over price and lead time. It is built on
native `asyncio` — NOT OS threads — so the Orchestrator can `await` it, genuinely suspending
and resuming per the §4.3 handoff contract, with deterministic (single-threaded) control
flow and no race conditions on the shared LedgerStore.

Design highlights (per Day-3 directive):
- **asyncio**: `run_supplier_negotiation` is an `async` coroutine the orchestrator awaits.
- **Dual-LLM personas**: a constrained Supplier-Persona replies "Accept" / "Counter" /
  "Reject" in natural language; our agent parses that messy NL into strict primitives.
- **Turn-Limited State Machine**: hard `max_turns=3`. No agreement by turn 3 -> autonomously
  sever, log, set `negotiation_status=FAILED`, and yield control back.
- **State Mutation Layer**: only parsed primitives (agreed price / lead time) are returned;
  ALL raw dialogue is routed to `incident_execution.log`, never into the ledger.
- **Pluggable vendor**: `VENDOR_MODE=llm` (default, live demo) vs `VENDOR_MODE=deterministic`
  (zero-cost offline testing/CI). Both exercise the same NL->primitives parsing path.
- **Isolated turn counter**: the sub-graph counts its OWN internal volleys; it MUST NEVER
  call `STORE.increment_loop()`. To the Orchestrator the whole negotiation is ONE loop.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Callable, Dict, Optional

from ledger_store import STORE, LedgerStore
from llm_utils import generate_with_retry
from guardrails import sanitize_write_payload, InjectionAttemptError, MAX_NL_STRING_LEN




# Scalar-only terms map. Nested objects are forbidden — every value must be a primitive.
TermsDict = Dict[str, str | int | float]

# The re-awaken callback the orchestrator may register (flat scalar payload).
NegotiationCallback = Callable[[str, Dict[str, str | int | float]], None]

# Hard ceiling on internal bargaining volleys (Turn-Limited State Machine). This is the
# sub-graph's OWN counter — completely separate from the ledger's loop_count budget.
MAX_TURNS = 3

# Vendor persona source: 'llm' (real Supplier-Persona LLM, default) or 'deterministic'
# (scripted mock for zero-cost offline tests). Live demo/submission uses 'llm'.
VENDOR_MODE = os.environ.get("VENDOR_MODE", "llm")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


# ---------------------------------------------------------------------------- #
# Supplier-Persona prompt — deliberately constrained to avoid politeness loops.
# ---------------------------------------------------------------------------- #
SUPPLIER_SYSTEM_PROMPT = """
You are SUPPLIER B, a procurement account manager for an APPROVED alternate vendor.
You are negotiating a rush order. Evaluate the buyer's offer STRICTLY on price and volume.

RULES:
- Your walk-away floor unit price is 45.00 USD. You will NOT sell below it.
- If the buyer's offered unit price >= 45.00, you ACCEPT.
- If it is below 45.00 but within ~8%, you COUNTER with a price at or just above your floor.
- If it is far below your floor, you REJECT.
- Reply in ONE or TWO short sentences of natural language.
- Your reply MUST begin with exactly one of these tokens: Accept / Counter / Reject.
- When you Accept or Counter, state the final unit price and lead time in days in prose.
Do NOT add pleasantries or ask follow-up questions. Be terse and decisive.
"""


def _init_genai_client() -> Optional[Any]:
    """Build the Gen AI client from GEMINI_API_KEY, or None (offline/no-key/no-sdk)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai  # unified google-genai SDK
    except ImportError:
        return None
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------- #
# Vendor reply generation (pluggable).
# ---------------------------------------------------------------------------- #
async def _vendor_reply(
    offer_price: float, offer_qty: int, turn: int, floor_price: float, lead_time_days: int
) -> str:
    """
    Produce the Supplier-Persona's natural-language reply to the buyer's offer.

    Returns messy NL text (e.g. "Counter: I can do 45.00 per unit with a 6 day lead time.")
    — the Negotiation Agent is responsible for parsing primitives out of it. The vendor's
    walk-away `floor_price` is per-supplier, so different vendors settle at different prices.
    """
    if VENDOR_MODE == "deterministic" or _init_genai_client() is None:
        return _deterministic_vendor_reply(offer_price, offer_qty, floor_price, lead_time_days)
    return await _llm_vendor_reply(offer_price, offer_qty, turn, floor_price, lead_time_days)


def _deterministic_vendor_reply(
    offer_price: float, offer_qty: int, floor_price: float, lead_time_days: int
) -> str:
    """Scripted vendor for zero-cost offline tests — mirrors the LLM persona's floor logic."""
    if offer_price >= floor_price:
        return f"Accept. We'll fulfill {offer_qty} units at {offer_price:.2f} USD each, {lead_time_days} day lead time."
    if offer_price >= floor_price * 0.92:
        # Counter at the floor.
        return f"Counter. The best I can do is {floor_price:.2f} USD per unit with a {lead_time_days} day lead time."
    return f"Reject. {offer_price:.2f} USD is far below our floor for {offer_qty} units."


async def _llm_vendor_reply(
    offer_price: float, offer_qty: int, turn: int, floor_price: float, lead_time_days: int
) -> str:
    """Invoke the real Supplier-Persona LLM (async) and return its raw NL reply."""
    client = _init_genai_client()
    from google.genai import types  # type: ignore

    # Inject this supplier's specific floor + lead time so distinct vendors quote distinctly.
    system_instruction = SUPPLIER_SYSTEM_PROMPT.replace("45.00", f"{floor_price:.2f}")
    prompt = (
        f"Negotiation turn {turn}. The buyer offers {offer_price:.2f} USD per unit "
        f"for {offer_qty} units on a rush order (your standard lead time is "
        f"{lead_time_days} days). Respond per your rules."
    )
    # Route through the SHARED resilient helper so the Supplier-Persona obeys the SAME
    # 429/5xx retry + daily-quota escalation policy as the Incident Commander. This is
    # CRITICAL under concurrent multi-vendor negotiation (asyncio.gather): firing several
    # simultaneous calls at one endpoint would otherwise trigger un-throttled 429 crashes.
    response = await generate_with_retry(
        client,
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
        ),
        source=f"negotiation:{floor_price:.2f}",
        log=LedgerStore.append_raw_log,
    )
    return response.text or "Reject."



# ---------------------------------------------------------------------------- #
# State Mutation Layer parsing: messy NL vendor reply -> strict primitives.
# ---------------------------------------------------------------------------- #
_PRICE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:USD|usd|per unit|/unit|each)")
_LEADTIME_RE = re.compile(r"([0-9]+)\s*[- ]?day")
_DECISION_RE = re.compile(r"\b(accept|counter|reject)\b", re.IGNORECASE)


def _parse_vendor_reply(raw_text: str, fallback_price: float, fallback_qty: int) -> TermsDict:
    """
    Extract strict primitives from the vendor's unstructured natural-language reply.

    This is the State Mutation Layer showcase: the messy NL never enters the ledger; only
    the parsed scalars (decision, unit price, lead time) do.
    """
    decision_match = _DECISION_RE.search(raw_text)
    decision = decision_match.group(1).upper() if decision_match else "REJECT"

    price_match = _PRICE_RE.search(raw_text)
    unit_price = round(float(price_match.group(1)), 2) if price_match else fallback_price

    lead_match = _LEADTIME_RE.search(raw_text)
    lead_time_days = int(lead_match.group(1)) if lead_match else 0

    return {
        "vendor_decision": decision,
        "agreed_unit_price_usd": unit_price,
        "agreed_lead_time_days": lead_time_days,
        "agreed_qty": fallback_qty,
    }


# ---------------------------------------------------------------------------- #
# The async negotiation sub-graph (Turn-Limited State Machine).
# ---------------------------------------------------------------------------- #
async def run_supplier_negotiation(
    incident_id: str,
    target_terms: TermsDict,
    *,
    supplier_id: str = "SUP-B",
    floor_price: float = 45.0,
    lead_time_days: int = 6,
    store: Optional[LedgerStore] = None,
    on_complete: Optional[NegotiationCallback] = None,
    write_status: bool = True,
) -> TermsDict:
    """
    Async LLM-to-LLM negotiation sub-graph (Section 4.3).

    The Orchestrator `await`s this coroutine: it writes `negotiation_status=IN_PROGRESS`,
    genuinely suspends while the bargaining volleys run, then resumes when this returns the
    flat scalar outcome. The internal `MAX_TURNS` counter is isolated — this coroutine NEVER
    calls STORE.increment_loop(); to the orchestrator the whole exchange is ONE loop.

    Args:
        incident_id: Incident this negotiation belongs to.
        target_terms: Desired terms, e.g.
            {"unit_price_ceiling_usd": 45.0, "required_lead_time_days": 6, "qty": 500}.
        supplier_id: The specific vendor being negotiated with (e.g. "SUP-B" / "SUP-C").
        floor_price: This supplier's walk-away price (distinct per vendor).
        lead_time_days: This supplier's quoted lead time (surfaced in the vendor reply).
        store: Ledger store to mutate (defaults to shared singleton).
        on_complete: Optional callback fired with the flat outcome to re-awaken the caller.
        write_status: When True (single negotiation), this coroutine writes
            `negotiation_status` itself. When False (CONCURRENT gather over multiple
            suppliers), the ORCHESTRATOR holds the master status lock and writes the final
            SUCCESS/FAILED once — so concurrent coroutines must NOT touch the shared field.

    Returns:
        Flat scalar outcome dict incl. `negotiation_outcome_status` (SUCCESS/FAILED),
        `agreed_supplier_id`, and the agreed primitives. Only primitives — ledger-safe.
    """
    active_store = store or STORE

    # Suspend point 1: mark IN_PROGRESS — but ONLY when we own the status lock. Under a
    # concurrent gather the orchestrator sets IN_PROGRESS once, so we skip it here.
    if write_status:
        active_store.mutate({"mitigation": {"negotiation_status": "IN_PROGRESS"}})
    LedgerStore.append_raw_log(
        "run_supplier_negotiation",
        f"incident_id={incident_id} supplier={supplier_id} floor={floor_price} "
        f"target_terms={target_terms} vendor_mode={VENDOR_MODE} dispatched",
    )

    # Buyer's opening position: start below the ceiling and step up toward it each turn.
    price_ceiling = float(target_terms.get("unit_price_ceiling_usd", 45.0))
    qty = int(target_terms.get("qty", 0))
    opening_offer = round(price_ceiling * 0.90, 2)  # open ~10% under the ceiling.

    outcome: TermsDict = {
        "negotiation_outcome_status": "FAILED",
        "agreed_supplier_id": supplier_id,
        "agreed_unit_price_usd": 0.0,
        "agreed_lead_time_days": 0,
        "agreed_qty": qty,
    }

    # --- Turn-Limited State Machine: at most MAX_TURNS bargaining volleys ---------------
    for turn in range(1, MAX_TURNS + 1):
        # Buyer steps its offer toward the ceiling across the allowed turns.
        offer_price = round(
            min(price_ceiling, opening_offer + (price_ceiling - opening_offer) * (turn - 1) / max(1, MAX_TURNS - 1)),
            2,
        )
        # Async (non-blocking) log write — inside the concurrent gather, a sync file write
        # would stall the event loop and serialize the vendors' LLM calls (see
        # LedgerStore.append_raw_log_async).
        await LedgerStore.append_raw_log_async(
            "negotiation_turn",
            f"incident_id={incident_id} supplier={supplier_id} turn={turn} buyer_offer={offer_price}",
        )

        raw_reply = await _vendor_reply(offer_price, qty, turn, floor_price, lead_time_days)
        # Route the messy raw dialogue to the isolated log — never into the ledger.
        await LedgerStore.append_raw_log_async(
            "negotiation_vendor_reply",
            f"incident_id={incident_id} supplier={supplier_id} turn={turn} raw={raw_reply!r}",
        )

        # SECURITY (defense-in-depth at the REAL injection surface, §5.1): the vendor's raw
        # NL reply is untrusted secondary-LLM output. Scan it BEFORE parsing. A hit is
        # non-crashing — we log it and treat this turn as a REJECT (skip to next volley),
        # so a malicious reply can neither drive parsing nor taint the ledger. (The parsed
        # `outcome` is scanned again below as a second layer.)
        try:
            # NL text -> use the looser natural-language length bound so a benign one-or-two
            # sentence reply isn't rejected on length; injection-pattern regex still applies.
            sanitize_write_payload({"raw_reply": raw_reply}, max_len=MAX_NL_STRING_LEN)
        except InjectionAttemptError as exc:

            await LedgerStore.append_raw_log_async(
                "negotiation_vendor_reply",
                f"RAW_REPLY_BLOCKED incident_id={incident_id} supplier={supplier_id} turn={turn} err={exc}",
            )
            await asyncio.sleep(0)
            continue

        parsed = _parse_vendor_reply(raw_reply, fallback_price=offer_price, fallback_qty=qty)


        if parsed["vendor_decision"] == "ACCEPT":
            outcome = {
                "negotiation_outcome_status": "SUCCESS",
                "agreed_supplier_id": supplier_id,
                "agreed_unit_price_usd": float(parsed["agreed_unit_price_usd"]),
                "agreed_lead_time_days": int(parsed["agreed_lead_time_days"]),
                "agreed_qty": qty,
            }
            break

        if parsed["vendor_decision"] == "COUNTER":
            # Accept the counter if it lands within our ceiling; else keep bargaining.
            counter_price = float(parsed["agreed_unit_price_usd"])
            if counter_price <= price_ceiling:
                outcome = {
                    "negotiation_outcome_status": "SUCCESS",
                    "agreed_supplier_id": supplier_id,
                    "agreed_unit_price_usd": counter_price,
                    "agreed_lead_time_days": int(parsed["agreed_lead_time_days"]),
                    "agreed_qty": qty,
                }
                break
            # otherwise continue to next turn (buyer will step price up)
        # REJECT -> continue to next turn; if turns exhausted, remains FAILED.

        # Cooperative yield so this is genuinely async (lets the event loop breathe).
        await asyncio.sleep(0)

    # SECURITY (defense-in-depth at the multi-agent boundary, §5.1): the outcome is derived
    # from an UNTRUSTED secondary LLM (the Supplier-Persona). Even though we parse it to
    # strict primitives, scan the whole outcome for injection/escape patterns BEFORE it can
    # flow back to the orchestrator and into STORE.mutate. A hit does NOT crash the vendor's
    # coroutine — we downgrade this vendor to a FAILED negotiation (log + drop), consistent
    # with the orchestrator's return_exceptions=True resilience, so a single poisoned reply
    # can never taint the ledger nor abort the concurrent gather.
    try:
        sanitize_write_payload(outcome)
    except InjectionAttemptError as exc:
        LedgerStore.append_raw_log(
            "run_supplier_negotiation",
            f"OUTCOME_BLOCKED incident_id={incident_id} supplier={supplier_id} err={exc}",
        )
        outcome = {
            "negotiation_outcome_status": "FAILED",
            "agreed_supplier_id": supplier_id,
            "agreed_unit_price_usd": 0.0,
            "agreed_lead_time_days": 0,
            "agreed_qty": qty,
        }

    # Forced resolution: commit the primitive status ONLY when we own the lock. Under a
    # concurrent gather the orchestrator writes the single final status after picking best.
    final_status = str(outcome["negotiation_outcome_status"])

    if write_status:
        active_store.mutate({"mitigation": {"negotiation_status": final_status}})
    LedgerStore.append_raw_log(
        "run_supplier_negotiation",
        f"incident_id={incident_id} supplier={supplier_id} resolved outcome={outcome}",
    )

    # Fire the re-awaken callback (flat scalar payload) if the caller registered one.
    if on_complete is not None:
        on_complete(incident_id, outcome)

    return outcome


__all__ = [
    "run_supplier_negotiation",
    "NegotiationCallback",
    "TermsDict",
    "MAX_TURNS",
    "VENDOR_MODE",
]
