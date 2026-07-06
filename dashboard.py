"""
dashboard.py
============
Streamlit presentation layer for the Autonomous Supply Chain Incident Commander — the
"Incident Command Center" cockpit.

This is the judge-facing window into an otherwise headless, autonomous agent. It is
deliberately a production-style operations CONSOLE, not a chatbot: the Incident Commander is
incident-triggered and self-driving, so a chat box would misrepresent it. It visualizes the
agent's live reasoning as a step-by-step activity log, with the Structured State Ledger
driving a business vitals strip and a plain-English resolution summary.

Three zones:
  * Zone 1 — VITALS strip: projected loss / revenue at risk / status.
  * Zone 2 — ACTIVITY LOG (the hero): each step rendered as a uniform, single-font line with
             a bold label (Reasoning / Action / Finding / Negotiation / Human Review). The
             guardrail step uses a green/red status box; the inline Approve/Reject buttons
             render here when a spend breach awaits a human verdict.
  * Zone 3 — RESOLUTION summary: a plain-English outcome with a green/red status label.

Design (why it's built this way):
- **Background agent thread.** Streamlit re-runs the whole script on every interaction; the
  agent is a long-running async loop. We run `asyncio.run(commander.run())` on a worker
  thread so the UI stays responsive. `ledger_store.STORE` is already thread-safe (RLock).
- **Observational `on_event` stream.** The orchestrator narrates every step through an
  `on_event(kind, message, data)` sink (purely observational — it never affects control
  flow). We append those events and render them as they arrive; in LIVE (Gemini) mode the
  real model latency naturally spaces the steps out (no artificial pacing).
- **ASYNC HITL bridge.** The orchestrator's `human_decision` runs INSIDE the agent thread's
  asyncio event loop. A blocking wait there would freeze that loop (and the concurrent
  negotiation sub-graph). We inject an ASYNC callback that `await`s an `asyncio.Event`; the
  UI thread wakes it via `loop.call_soon_threadsafe`.
- **Live/Offline toggle.** The free Gemini tier is ~20 req/day; rehearsing on the live model
  would exhaust quota. The sidebar defaults to the deterministic OFFLINE planner and flips to
  LIVE Gemini only for the final take. The engaged core is shown in the header so it's never
  ambiguous which reasoning core actually ran.

Run it with:  `streamlit run dashboard.py`
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from ledger_store import STORE, LedgerStore
from orchestrator import (
    IncidentCommander,
    GEMINI_MODEL,
    REVENUE_AT_RISK_USD,
    INTERNAL_TRANSFER_SURPLUS_UNITS,
    AIR_FREIGHT_CAPACITY_UNITS,
)
from guardrails import SpendAuthorityResult, SPEND_AUTHORITY_LIMIT_USD



# An event is a (kind, human_readable_message, data) triple emitted by the orchestrator.
EventTuple = Tuple[str, str, Dict[str, Any]]

# Bold label per event `kind`. Production-friendly wording (NOT internal ReAct jargon) in
# consistent title case. We deliberately avoid emojis in the log text; the only status glyphs
# live on the guardrail line (green/red box) and the resolution summary.
_STEP_LABEL: Dict[str, str] = {
    "thought": "Reasoning",
    "action": "Action",
    "observation": "Finding",
    "negotiation": "Negotiation",
    "hitl": "Human Review",
}

# Descriptive, business-friendly purpose for each tool the agent can call. The Action line
# reads "<purpose> — calling `<tool>`" so a judge understands WHY the agent invoked it, not
# just the raw function name. Deterministic (works identically in live and offline mode).
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


# CSS to present a clean, production-style console: hide Streamlit's developer chrome (top

# toolbar / Deploy button, hamburger main menu, and the "Made with Streamlit" footer). These
# add zero value to an enterprise agent demo and look like a dev sandbox otherwise.
_HIDE_CHROME_CSS = """
<style>
[data-testid="stToolbar"] { visibility: hidden; height: 0; position: fixed; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
"""


def _md(text: str) -> str:
    """Escape characters Streamlit markdown would misinterpret.

    CRITICAL: Streamlit renders `$...$` as LaTeX math. Our narration is full of dollar
    amounts (e.g. "at $0 external spend … loss of $20,034"), so an unescaped pair of `$`
    would swallow the text between them into a serif math font AND drop the currency symbols.
    Escaping every `$` as `\\$` keeps the money literal and the font uniform.
    """
    return text.replace("$", "\\$")


# Inline "green circle with a white check" badge — a real filled circle (border-radius 50%)
# with a centered ✓, rendered via HTML so we control the exact look (st.status can only show
# its own flat glyph). Prepended to every COMPLETED step/negotiation header. Kept tiny and
# vertically-aligned so it sits neatly before the title text.
_DONE_BADGE = (
    "<span style='display:inline-flex;align-items:center;justify-content:center;"
    "width:1.15em;height:1.15em;border-radius:50%;background:#22c55e;color:#fff;"
    "font-size:0.8em;font-weight:700;line-height:1;vertical-align:middle;"
    "margin-right:0.45em;'>✓</span>"
)


def _done_header(title: str) -> str:
    """A completed-step header: green-circle-check badge + bold title (HTML, `$` escaped)."""
    return f"{_DONE_BADGE}<b>{_md(title)}</b>"



# --------------------------------------------------------------------------- #
# Async HITL bridge — the safe cross-thread pause/resume for the guardrail.
# --------------------------------------------------------------------------- #
class HITLBridge:
    """Async human-decision provider that pauses the agent's event loop for a UI verdict.

    `decide` runs on the AGENT thread's event loop (the orchestrator awaits it). It records
    the breach details, then `await`s an `asyncio.Event` — yielding control to the loop
    (never blocking it). The UI thread renders Approve/Reject buttons; clicking one calls
    `resolve`, which wakes the coroutine via `loop.call_soon_threadsafe` (the ONLY
    thread-safe way to signal an asyncio primitive from another thread).
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._event: Optional[asyncio.Event] = None
        # Read by the UI thread each rerun (plain attribute reads — atomic enough for a demo).
        self.pending: bool = False
        self.result: Optional[SpendAuthorityResult] = None
        self._decision: Optional[bool] = None

    async def decide(self, result: SpendAuthorityResult) -> bool:
        """Awaited by the orchestrator when a spend breach needs a human verdict."""
        self._loop = asyncio.get_running_loop()
        self._event = asyncio.Event()
        self._decision = None
        self.result = result
        self.pending = True
        try:
            await self._event.wait()  # yields to the loop; UI thread will set() it
        finally:
            self.pending = False
        return bool(self._decision)

    def resolve(self, approved: bool) -> None:
        """Called FROM THE UI THREAD when the operator clicks Approve/Reject."""
        self._decision = approved
        loop, event = self._loop, self._event
        if loop is not None and event is not None:
            loop.call_soon_threadsafe(event.set)

    def reset(self) -> None:
        self._loop = None
        self._event = None
        self.pending = False
        self.result = None
        self._decision = None


# --------------------------------------------------------------------------- #
# Per-session shared state (survives Streamlit reruns via st.session_state).
# The background thread mutates the plain attributes below directly (NOT via the
# st.session_state API, which is not safe to touch from a non-UI thread).
# --------------------------------------------------------------------------- #
@dataclass
class AgentSession:
    bridge: HITLBridge = field(default_factory=HITLBridge)
    thread: Optional[threading.Thread] = None
    running: bool = False
    started: bool = False
    error: Optional[str] = None
    # The full ordered stream of orchestrator events (appended by the agent thread).
    events: List[EventTuple] = field(default_factory=list)
    # Which reasoning core actually engaged this run ("LIVE · <model>" or "OFFLINE …").
    core_mode: str = ""
    # The (order_qty, delay) the currently-displayed run used — so the UI can auto-clear a
    # stale flow when the operator changes an input without pressing Reset first.
    run_params: Tuple[int, int] = (0, 0)
    # Last STORE revision the UI observed — used by the stale-snapshot convergence guard.
    last_revision: int = -1
    # Latched terminal outcome ("resolved" | "escalated"), set EXACTLY ONCE from the
    # orchestrator's final resolution event when the run settles. Monotonic: once set it never
    # flips, so the vitals + Zone-3 label can never flicker between Escalated and Resolved
    # during the brief post-HITL settling window (the earlier bug). None until settled.
    outcome: Optional[str] = None





def _on_event(sess: AgentSession):
    """Build the orchestrator `on_event` callback that buffers the live step stream.

    Runs on the AGENT thread. We only append to a plain list (thread-safe enough for a demo
    — appends are atomic under CPython's GIL) and never touch the st.session_state API here.
    """

    def _cb(kind: str, message: str, data: Dict[str, Any]) -> None:
        sess.events.append((kind, message, data))
        # Keep the buffer bounded so a long run can't grow memory without bound.
        if len(sess.events) > 500:
            del sess.events[:-500]

    return _cb


def _run_agent(sess: AgentSession, params: Dict[str, Any]) -> None:
    """Background-thread entry point: configure the run, then drive the async loop.

    Runs entirely off the Streamlit UI thread. Sets the reasoning-core / vendor mode via
    env (read at commander construction), initializes the incident, then `asyncio.run`s the
    orchestrator to completion. The original API key is passed IN (captured in session_state
    on the UI thread) and `os.environ` is restored in `finally` so a prior OFFLINE run can
    never clobber the key for a later LIVE run.
    """
    saved_key = os.environ.get("GEMINI_API_KEY", "")
    saved_vendor = os.environ.get("VENDOR_MODE", "")
    try:
        # --- Reasoning core selection (Offline default protects the free quota). --------
        if params["offline"]:
            os.environ["GEMINI_API_KEY"] = ""            # -> deterministic offline planner
            os.environ["VENDOR_MODE"] = "deterministic"  # -> scripted vendor (no LLM calls)
        else:
            os.environ["GEMINI_API_KEY"] = params["gemini_key"]  # -> live Gemini core
            os.environ["VENDOR_MODE"] = "llm"

        # NOTE: the incident ledger is initialized SYNCHRONOUSLY on the UI thread in
        # `_start` (before this thread is spawned), so it is guaranteed to exist by the time
        # the UI reads `STORE.snapshot()`. We do NOT init it here — doing so on the worker
        # thread created an init race the UI could only paper over with an exception + sleep.
        commander = IncidentCommander(

            order_quantity=params["order_quantity"],
            spend_authority_limit_usd=params["spend_limit"],
            human_decision=sess.bridge.decide,  # ASYNC bridge — awaited by the orchestrator
            on_event=_on_event(sess),           # observational step stream -> Zone 2 log
        )
        # Record which core actually engaged so the header can show it truthfully (and the
        # operator knows when a run spent live quota). Derived from the real client bootstrap.
        sess.core_mode = (
            f"LIVE · {GEMINI_MODEL}" if commander.llm_enabled else "OFFLINE · deterministic planner"
        )
        asyncio.run(commander.run(verbose=False))
    except Exception as exc:  # surface any failure to the UI instead of dying silently
        sess.error = repr(exc)
        LedgerStore.append_raw_log("dashboard", f"AGENT_THREAD_ERROR {exc!r}")
    finally:
        # Restore the process env so mode selection never leaks between runs (this is what
        # previously broke the LIVE toggle after an OFFLINE run cleared the key).
        os.environ["GEMINI_API_KEY"] = saved_key
        os.environ["VENDOR_MODE"] = saved_vendor
        sess.running = False


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
def _get_session() -> AgentSession:
    if "agent_session" not in st.session_state:
        st.session_state.agent_session = AgentSession()
    return st.session_state.agent_session


def _get_original_key() -> str:
    """Capture the real GEMINI_API_KEY ONCE in session_state.

    Streamlit re-executes the whole script every rerun, and a prior OFFLINE run sets
    os.environ["GEMINI_API_KEY"]="" — so a module-level capture would read back "" and
    silently disable the LIVE toggle. Storing it in session_state (which persists across
    reruns) makes the captured key immune to that env-clobber.
    """
    if "original_gemini_key" not in st.session_state:
        st.session_state.original_gemini_key = os.environ.get("GEMINI_API_KEY", "")
    return st.session_state.original_gemini_key


def _start(sess: AgentSession, params: Dict[str, Any]) -> None:
    """Spawn the agent thread ONCE (guard against Streamlit's per-interaction reruns).

    INIT ORDER (fixes the init race): the incident ledger is initialized SYNCHRONOUSLY here,
    on the UI thread, BEFORE the worker thread is spawned. This guarantees `STORE.snapshot()`
    always succeeds on the next rerun — so the UI needs no exception-handler-plus-sleep hack
    to tolerate an uninitialized ledger. `init_incident` is a fast, lock-protected in-memory
    call, so running it on the UI thread is safe and cheap.
    """
    if sess.running:
        return
    sess.bridge.reset()
    sess.events.clear()
    sess.core_mode = ""
    sess.last_revision = -1
    sess.error = None

    # Establish the single source of truth up front (UI thread) — the worker thread will only
    # read/mutate it, never (re-)create it. The agent's strategy EMERGES from comparing the
    # order quantity against the finite resources (surplus 350 / air capacity 420); the delay
    # independently drives the dynamic projected loss (no scenario switch steers the outcome).
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
        replacement_order_qty=params["order_quantity"],
        delay_days=params["delay_days"],
    )

    # Remember the (qty, delay) this run used, so the UI can auto-clear a stale flow if the
    # operator changes an input without pressing Reset.
    sess.run_params = (int(params["order_quantity"]), int(params["delay_days"]))


    sess.running = True
    sess.started = True
    sess.thread = threading.Thread(target=_run_agent, args=(sess, params), daemon=True)
    sess.thread.start()




def _reset(sess: AgentSession) -> None:
    """Reset the UI session. (A live thread is daemon; we just detach and start fresh.)"""
    st.session_state.agent_session = AgentSession()


def _thread_alive(sess: AgentSession) -> bool:
    return sess.thread is not None and sess.thread.is_alive()


# --------------------------------------------------------------------------- #
# Zone renderers
# --------------------------------------------------------------------------- #
def _render_vitals(ledger: Any, running: bool, outcome: Optional[str]) -> None:
    """Zone 1 — the business vitals strip. Reads the live ledger (single source of truth).

    Deliberately shows BUSINESS metrics only (loss / revenue / status). The internal ReAct
    loop counter is NOT surfaced — a real operations console reports outcomes, not the
    agent's internal iteration mechanics.
    """
    metrics = ledger.metrics

    # Status is derived from the LATCHED `outcome` (set once when the run settles), NEVER
    # re-derived from the live ledger — that is what killed the Escalated↔Resolved flicker:
    # while running we show "In progress"; once settled we show the single, monotonic terminal
    # label. `outcome` can only ever be set to one value for a given run, so it cannot flip.
    if running or outcome is None:
        status_label = "In progress" if running else "Standby"
        status_help = "Agent is working the incident" if running else "Awaiting dispatch"
    elif outcome == "escalated":
        status_label, status_help = "Escalated", "Guardrail breached — handed to a human"
    else:  # "resolved"
        status_label, status_help = "Resolved", "Incident mitigated autonomously"

    c1, c2, c3 = st.columns(3)

    c1.metric("Projected Loss", f"${metrics.projected_total_loss_usd:,.0f}")
    c2.metric("Revenue at Risk", f"${metrics.revenue_at_risk_usd:,.0f}")
    c3.metric("Status", status_label, help=status_help)


def _thought_text(message: str) -> str:
    """Reasoning body, with a neutral fallback when a live model omits the Thought line."""
    body = str(message).strip()
    return body or "Assessing the incident and selecting the next action."


def _action_text(message: str, data: Dict[str, Any]) -> str:
    """Descriptive Action line: '<purpose> — calling <tool>' (falls back to the tool name).

    The orchestrator emits the raw tool name as the action message; we translate it into a
    business-friendly purpose so a judge sees WHY the tool was called, not just its name.
    """
    tool = str(data.get("tool") or message).strip()
    purpose = _TOOL_PURPOSE.get(tool)
    if purpose:
        return f"{purpose} — calling 🔧 `{tool}` tool."
    return f"Calling 🔧 `{tool}` tool."



def _render_react_card(steps: List[EventTuple]) -> None:
    """Render one COMPLETED ReAct turn (Reasoning → Action → Finding) as a static block.

    DESIGN (why NOT st.status): st.status widgets re-mount on every one of our ~0.4s
    auto-reruns, which restarts their spinner and causes visible flicker; their label is also
    plain text, so we can't render a custom green-circle tick inside it. Instead we render a
    STATIC header — a green-circle-white-check badge (`_done_header`) + the step title — with
    the Reasoning / Action / Finding detail tucked into a COLLAPSED `st.expander`. Static
    markdown + a plain expander don't re-mount/animate on reruns, so there is ZERO flicker,
    and the tick is exactly the circled green check requested. The single live "reasoning…"
    spinner (shown separately at the tail of the log while the thread is alive) is the honest
    in-progress indicator — the real LLM latency happens BETWEEN cards, not within one.
    """
    # Card title = the tool/action purpose if known, else a generic "Reasoning step".
    title = "Reasoning step"
    for k, m, d in steps:
        if k == "action":
            tool = str(d.get("tool") or m).strip()
            title = _TOOL_PURPOSE.get(tool, f"Calling {tool}")
            break

    st.markdown(_done_header(title), unsafe_allow_html=True)
    with st.expander("Details", expanded=False):
        for k, m, d in steps:
            if k == "thought":
                st.markdown(f"**Reasoning** — {_md(_thought_text(m))}")
            elif k == "action":
                st.markdown(f"**Action** — {_md(_action_text(m, d))}")
            elif k == "observation":
                body = str(m).strip()
                if body:
                    st.markdown(f"**Finding** — {_md(body)}")



def _render_negotiation(steps: List[EventTuple], is_last: bool, running: bool) -> None:
    """Render the supplier negotiation block (same static treatment as the ReAct cards).

    While the sub-graph is still bargaining (agent running, this is the tail block, and no
    winning quote has landed yet) we show a live "Negotiating with SUP-B and SUP-C…" spinner
    via `st.status(state="running")`. Once the "Best quote secured" line arrives — or the run
    ends — it renders as a STATIC "Negotiation successful" header with the same green-circle
    check badge as the cards, and the buyer<->supplier chat transcript inside a COLLAPSED
    expander. (Static-once-done avoids the st.status re-mount flicker; the transient spinner
    only appears during the genuinely-in-progress LIVE window.)
    """
    # Discover the vendors being negotiated with (from the opening system event's `vendors`
    # datum; fall back to the distinct supplier ids seen in the transcript).
    vendors = ""
    for _k, _m, data in steps:
        if data.get("vendors"):
            vendors = str(data["vendors"])
            break
    if not vendors:
        seen = [str(d["supplier"]) for _k, _m, d in steps if d.get("supplier")]
        # de-dupe preserving order
        vendors = ", ".join(dict.fromkeys(seen))
    vendors_phrase = vendors.replace(", ", " and ") if vendors else "alternate suppliers"

    # Done when the winning-quote system line has arrived (or the run has otherwise ended).
    done = any(str(m).startswith("Best quote secured") for _k, m, _d in steps) or not (
        running and is_last
    )

    def _emit_transcript() -> None:
        for _k, message, data in steps:
            role = str(data.get("role", "system"))
            # Skip the orchestrator's opening "Negotiating concurrently with N alternate
            # suppliers…" framing line — it's redundant once the header states the outcome.
            if role == "system" and str(message).startswith("Negotiating concurrently"):
                continue
            if role == "buyer":
                with st.chat_message("user", avatar="🧑‍💼"):
                    st.markdown(_md(message))
            elif role == "supplier":
                sid = str(data.get("supplier", "Supplier"))
                with st.chat_message("assistant", avatar="🏭"):
                    st.markdown(_md(f"**{sid}:** {message}"))
            else:
                # System framing lines (bidding open / best quote) as a plain notice.
                st.markdown(_md(message))

    if not done:
        # Genuinely in-progress (LIVE bargaining window). Render a PLAIN STATIC line — NOT an
        # st.status widget, which re-mounts on every 0.4s rerun and flickers. Plain markdown
        # re-prints identically each rerun, so it's flicker-free; the single tail
        # "Agent is reasoning…" spinner provides the live motion cue.
        st.markdown(f"**Negotiating with {vendors_phrase}…**")
        return

    # Completed: STATIC green-circle header + collapsed transcript (no flicker on reruns).
    st.markdown(_done_header("Negotiation successful"), unsafe_allow_html=True)
    with st.expander("View negotiation transcript", expanded=False):
        _emit_transcript()






def _render_activity_log(sess: AgentSession) -> None:
    """Zone 2 — the activity log (the hero of the console).

    Consecutive Reasoning → Action → Finding events are grouped into a single status card
    (spinner while processing, green check when done). Negotiation events are collected and
    rendered together as a supplier chat. Guardrail lines keep their green/red status box.
    """
    st.subheader("Agent activity")

    if not sess.events:
        st.caption("Awaiting the agent's first step…")
        return

    # Partition the flat event stream into ordered render blocks:
    #   ("card", [thought/action/observation events])  — one ReAct turn
    #   ("negotiation", [negotiation events])           — the supplier chat
    #   ("guardrail", event)                             — a single guardrail line
    blocks: List[Tuple[str, Any]] = []
    card: List[EventTuple] = []
    negotiation: List[EventTuple] = []

    def _flush_card() -> None:
        if card:
            blocks.append(("card", list(card)))
            card.clear()

    def _flush_negotiation() -> None:
        if negotiation:
            blocks.append(("negotiation", list(negotiation)))
            negotiation.clear()

    for kind, message, data in sess.events:
        if kind in ("thought", "action", "observation"):
            _flush_negotiation()
            # A new 'thought' starts a fresh card.
            if kind == "thought" and card:
                _flush_card()
            card.append((kind, message, data))
        elif kind == "negotiation":
            _flush_card()
            negotiation.append((kind, message, data))
        elif kind == "guardrail":
            _flush_card()
            _flush_negotiation()
            blocks.append(("guardrail", (kind, message, data)))
        elif kind == "hitl":
            _flush_card()
            _flush_negotiation()
            blocks.append(("hitl", (kind, message, data)))
        # 'resolution' is handled by Zone 3 — ignore here.
    _flush_card()
    _flush_negotiation()

    for i, (block_kind, payload) in enumerate(blocks):
        is_last = i == len(blocks) - 1
        if block_kind == "card":
            _render_react_card(payload)
        elif block_kind == "negotiation":
            _render_negotiation(payload, is_last, sess.running)

        elif block_kind == "guardrail":
            _k, message, data = payload
            if data.get("passed"):
                st.success(_md(message))
            else:
                st.error(_md(message))
        elif block_kind == "hitl":
            _k, message, _data = payload
            st.markdown(f"**Human Review** — {_md(str(message))}")

    # LIVE "working" indicator: a single spinner at the TAIL of the log while the agent
    # thread is still alive. This is the honest in-progress cue — the real LLM latency happens
    # BETWEEN steps (inside _reason, before the next card exists), so a per-card spinner can
    # never show; one tail spinner tied to `running` captures that dead-air correctly and,
    # being a single element, cannot flicker. (Offline finishes in ~ms so it's rarely seen.)
    if sess.running and not (sess.bridge.pending and sess.bridge.result is not None):
        st.status("🧠 Agent is reasoning…", state="running", expanded=False)


    # Inline HITL approval gate — rendered at the tail of the log when a spend breach is
    # actively awaiting a human verdict (the agent's event loop is paused on the bridge).
    if sess.bridge.pending and sess.bridge.result is not None:
        r = sess.bridge.result
        st.divider()
        st.warning(
            _md(
                f"Human decision required — spend ${r.spend_usd:,.0f} exceeds the delegated "
                f"authority ${r.limit_usd:,.0f} for supplier {r.supplier_id}."
            )
        )
        h1, h2 = st.columns(2)
        if h1.button("Approve over-limit spend", use_container_width=True, type="primary"):
            sess.bridge.resolve(True)
            st.rerun()
        if h2.button("Reject", use_container_width=True):
            sess.bridge.resolve(False)
            st.rerun()



def _render_resolution(sess: AgentSession, outcome: Optional[str]) -> None:
    """Zone 3 — plain-English resolution summary with a thin green/red status label.

    Uses the orchestrator's streamed `resolution` event text for the body, and the LATCHED
    `outcome` ("resolved"/"escalated") for the color label — NOT a fresh ledger read — so the
    label is monotonic and can never flip during the post-approve settling window. The body is
    rendered as normal markdown (unified font, `$` escaped); the color cue is a compact status
    label above it (not a full alert box) so the font matches the log.
    """
    resolution_msg: Optional[str] = None
    for kind, message, _data in reversed(sess.events):
        if kind == "resolution" and message != "Incident assessment complete.":
            resolution_msg = message
            break

    escalated = outcome == "escalated"

    if resolution_msg is None:
        st.caption("The resolution summary will appear here once the agent concludes.")
        return

    # Strip any leading status glyph the orchestrator prepended; the color label carries it.
    body = resolution_msg
    for glyph in ("✅", "🚨"):
        if body.startswith(glyph):
            body = body[len(glyph):].strip()

    st.subheader("Resolution")
    if escalated:
        st.markdown(":red[**● ESCALATED**]")
    else:
        st.markdown(":green[**● RESOLVED**]")
    st.markdown(_md(body))



def main() -> None:
    st.set_page_config(page_title="Incident Command Center", layout="wide")
    st.markdown(_HIDE_CHROME_CSS, unsafe_allow_html=True)

    sess = _get_session()
    original_key = _get_original_key()

    # ----------------------------- Sidebar controls ----------------------------------- #
    with st.sidebar:
        st.header("Incident Controls")
        offline = st.toggle(
            "Offline reasoning (deterministic, free)",
            value=True,
            help="ON = zero-cost deterministic planner for rehearsals. OFF = live Gemini "
            "(uses your API quota). Default ON to protect the free tier.",
            disabled=sess.running,
        )
        # Live-mode requires a Gemini API key. Surface that requirement inline the moment the
        # operator turns Offline OFF, so it's obvious a key must be present in the .env — and
        # that the run gracefully falls back to the deterministic planner if it's missing.
        if not offline:
            st.caption(
                "Live mode needs `GEMINI_API_KEY` in `.env`; without it, it falls back to "
                "the offline planner."
            )


        # ORDER QUANTITY — the single lever the agent's strategy EMERGES from (no scenario
        # switch). It is compared against the fixed resources below; a bigger order closes
        # cheaper options and pushes the agent up the escalation ladder.
        order_quantity = st.number_input(
            "Order quantity (units)",
            min_value=1,
            max_value=100000,
            value=500,
            step=10,
            help=(
                f"The agent's strategy emerges from this vs the incident resources: "
                f"≤{INTERNAL_TRANSFER_SURPLUS_UNITS} → internal transfer (no spend); "
                f"≤{AIR_FREIGHT_CAPACITY_UNITS} → air-freight expedite; "
                f">{AIR_FREIGHT_CAPACITY_UNITS} → alternate supplier + negotiation. It also "
                f"sets total spend vs the ${SPEND_AUTHORITY_LIMIT_USD:,.0f} authority."
            ),
            disabled=sess.running,
        )

        # SHIPMENT DELAY — the second, independent lever. Drives the DYNAMIC projected loss
        # (bigger delay => bigger exposure), decoupled from the strategy choice.
        delay_days = st.number_input(
            "Shipment delay (days)",
            min_value=0,
            max_value=60,
            value=9,
            step=1,
            help="How many days the shipment has slipped. Drives the projected financial "
            "loss (penalty accrual + post-buffer downtime) — a bigger delay means a bigger "
            "exposure. Independent of which mitigation the agent selects.",
            disabled=sess.running,
        )

        # The spend-authority limit is a FIXED delegated signing authority — in a real

        # enterprise it's set by senior management and locked, not re-tuned per incident. So
        # we display it as a read-only notice rather than an editable field; the constant is
        # passed straight through to the agent.
        st.caption(
            f"Spend authority: **${SPEND_AUTHORITY_LIMIT_USD:,.0f}** — fixed delegated limit; "
            "spend above this escalates to a human."
        )

        col_a, col_b = st.columns(2)
        if col_a.button("Start", disabled=sess.running, use_container_width=True, type="primary"):
            _start(
                sess,
                {
                    "offline": offline,
                    "order_quantity": int(order_quantity),
                    "delay_days": int(delay_days),
                    "spend_limit": float(SPEND_AUTHORITY_LIMIT_USD),
                    "gemini_key": original_key,
                },
            )

        if col_b.button("Reset", disabled=sess.running, use_container_width=True):
            _reset(sess)
            st.rerun()

        requested = "OFFLINE (deterministic)" if offline else f"LIVE · {GEMINI_MODEL}"
        st.caption(f"Selected core: **{requested}**")
        if not offline and not original_key:
            st.warning("No API key found — a live run will fall back to the offline planner.")

    # AUTO-CLEAR STALE FLOW: if a previous run is displayed (started, not running) and the
    # operator changes the order quantity or delay without pressing Reset, wipe the old flow
    # so the UI cleanly returns to the "press Start" state for the new inputs — otherwise the
    # old run's cards/chat linger and visually fight the next run. Only fires between runs
    # (the inputs are disabled while running), so it's safe.
    current_params = (int(order_quantity), int(delay_days))
    if (
        sess.started
        and not sess.running
        and not _thread_alive(sess)
        and current_params != sess.run_params
    ):
        _reset(sess)
        st.rerun()


    # ------------------------------- Header -------------------------------------------- #

    st.title("Autonomous Supply Chain Incident Commander")
    st.caption(
        "An autonomous, incident-triggered agent that reasons over a structured state "
        "ledger to resolve a critical procurement disruption — with a hard financial "
        "guardrail and human-in-the-loop escalation."
    )

    if not sess.started:
        st.info(
            "Set the **order quantity** and **shipment delay** in the sidebar, then press "
            "**Start**. The agent chooses its own mitigation from the order size: try "
            "**≤ 300** for an autonomous internal-transfer resolution, or **≥ 500** to force "
            "an alternate-supplier purchase that breaches spend authority and triggers "
            "**human review**."
        )
        return



    if sess.error:
        st.error(f"Agent error: {sess.error}")

    # Read the live ledger (thread-safe). It was initialized SYNCHRONOUSLY in `_start` before
    # the worker thread was spawned, so once `sess.started` is true the ledger always exists —
    # no exception-handler + sleep hack needed for cross-thread init timing.
    ledger = STORE.snapshot()

    ctx = ledger.context

    meta = ledger.metadata
    core_note = f" · Core: {sess.core_mode}" if sess.core_mode else ""
    st.caption(
        f"Incident `{meta.id}` · SKU **{ctx.target_sku}** · Supplier {ctx.primary_supplier_id} "
        f"· Severity {meta.severity}{core_note}"
    )

    # LATCH THE TERMINAL OUTCOME ONCE (flicker fix): the moment the run has genuinely settled
    # (thread dead), record "resolved"/"escalated" a SINGLE time from the orchestrator's final
    # `resolution` event (which carries a `resolved` bool) — never re-derived from the live
    # ledger. Because it's written once and then frozen, the vitals + Zone-3 label can't flip
    # between Escalated and Resolved during the post-approve settling repaints.
    finished = not sess.running and not _thread_alive(sess)
    if finished and sess.outcome is None:
        resolved = True
        for kind, message, data in reversed(sess.events):
            if kind == "resolution" and message != "Incident assessment complete.":
                resolved = bool(data.get("resolved", not ledger.status.guardrail_status == "BREACHED"))
                break
        else:
            # No resolution event captured — fall back to the settled ledger state.
            resolved = ledger.status.guardrail_status != "BREACHED"
        sess.outcome = "resolved" if resolved else "escalated"

    # ------------------------------- Zone 1: vitals ------------------------------------ #
    _render_vitals(ledger, sess.running, sess.outcome)
    st.divider()

    # ------------------------------ Zone 2: activity log ------------------------------- #
    _render_activity_log(sess)

    # ------------------------------- Zone 3: resolution -------------------------------- #
    # Only shown once the incident has SETTLED (resolved/escalated) — including the State
    # Ledger JSON, which is the concrete "single source of truth" architecture proof judges
    # can inspect. Kept out of the live view (mid-run JSON is noise) and surfaced only at the
    # end as an audit artifact.
    if finished:
        st.divider()
        _render_resolution(sess, sess.outcome)

        with st.expander("State Ledger (JSON) — single source of truth"):
            st.json(ledger.model_dump(mode="json"))


    # ------------------------------- Live refresh loop --------------------------------- #
    # Keep repainting while the agent is working or a HITL decision is pending, plus extra
    # passes until the store revision stabilizes. This is the STALE-SNAPSHOT FIX: only when
    # the thread is dead AND the revision has settled do we stop — so the vitals converge to
    # the true final numbers instead of freezing on an early $0 snapshot.
    revision_advancing = STORE.revision != sess.last_revision
    sess.last_revision = STORE.revision

    keep_going = (
        sess.running
        or _thread_alive(sess)
        or sess.bridge.pending
        or revision_advancing
    )
    if keep_going:
        time.sleep(0.4)
        st.rerun()


if __name__ == "__main__":
    main()
