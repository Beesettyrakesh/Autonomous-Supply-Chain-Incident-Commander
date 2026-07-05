"""
guardrails.py
=============
Deterministic, code-enforced safety barriers (Section 5). The LLM has NO control over any
of these — they intercept the execution path AFTER the State Mutation Layer commits a
change, exactly as the architecture requires.

This module is intentionally isolated (pure functions + small dataclasses, no orchestrator
imports) so every barrier is independently unit-testable and reusable by the CLI and the
Streamlit dashboard.

Barriers implemented here:
  1. Financial Spend-Authority Guardrail (§5.2)
     - Models a procurement manager's FIXED delegated approval authority.
     - The realistic per-incident variable is the ORDER QUANTITY (hence total spend), not
       the limit. Spend <= limit -> agent may auto-approve; spend > limit -> pause for a
       HUMAN-IN-THE-LOOP (HITL) approve/reject decision.
  2. Jailbreak / Injection Sanitization (§5.1)
     - A regex + length-boundary scan applied to any parameter destined for a write action
       (e.g. committing a strategy / persisting negotiated terms). A hit ABORTS the write.

Configuration:
  - The spend-authority limit is FIXED policy (default $20,000) but overridable via the
    `SPEND_AUTHORITY_LIMIT_USD` environment variable so judges can retune it for a demo
    without code edits. The realistic demo variable remains the order quantity.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, cast



# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _load_spend_authority_limit() -> float:
    """Read the delegated spend-authority limit (USD). Fixed policy; env-overridable."""
    raw = os.environ.get("SPEND_AUTHORITY_LIMIT_USD", "20000")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 20000.0


# Delegated approval authority: spend at or below this may be auto-approved by the agent;
# spend above it must be escalated to a human (HITL). A procurement manager's signing limit.
SPEND_AUTHORITY_LIMIT_USD: float = _load_spend_authority_limit()


# --------------------------------------------------------------------------- #
# 1. Financial Spend-Authority Guardrail (§5.2)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpendAuthorityResult:
    """Outcome of a spend-authority check — flat, ledger-safe primitives only."""

    within_authority: bool          # True -> agent may auto-approve; False -> needs HITL
    spend_usd: float                # the evaluated total spend (unit price * qty)
    limit_usd: float                # the delegated authority limit checked against
    supplier_id: str                # the vendor the spend is with
    reason: str                     # human-readable explanation (for the escalation record)


def check_spend_authority(
    supplier_id: str,
    unit_price_usd: float,
    quantity: int,
    limit_usd: float = SPEND_AUTHORITY_LIMIT_USD,
) -> SpendAuthorityResult:
    """
    Evaluate whether a negotiated purchase is within the agent's delegated spend authority.

    Wraps `decision_helpers.policy_check` (approved-vendor + spend-cap compliance) so there
    is a single source of truth for the compliance rule, and enriches the boolean into a
    structured result the orchestrator can act on and log.

    Args:
        supplier_id: The winning vendor the purchase order would be placed with.
        unit_price_usd: The negotiated per-unit price.
        quantity: The order quantity (the realistic per-incident variable).
        limit_usd: The delegated authority limit (defaults to the configured policy limit).

    Returns:
        SpendAuthorityResult with `within_authority` True (auto-approve) or False (HITL).
    """
    # Imported locally to keep this module free of heavy import-time dependencies.
    from decision_helpers import policy_check

    spend_usd = round(float(unit_price_usd) * int(quantity), 2)
    within = policy_check(supplier_id, spend_usd, per_transaction_cap_usd=limit_usd)

    if within:
        reason = (
            f"spend_within_authority: ${spend_usd:,.2f} <= ${limit_usd:,.2f} "
            f"(supplier {supplier_id}) — auto-approved"
        )
    elif supplier_id not in {"SUP-A", "SUP-B", "SUP-C"}:
        reason = f"unapproved_vendor: {supplier_id} is not an approved supplier"
    else:
        reason = (
            f"spend_exceeds_authority: ${spend_usd:,.2f} > ${limit_usd:,.2f} "
            f"(supplier {supplier_id}) — human approval required"
        )

    return SpendAuthorityResult(
        within_authority=within,
        spend_usd=spend_usd,
        limit_usd=round(float(limit_usd), 2),
        supplier_id=supplier_id,
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# 2. Jailbreak / Injection Sanitization (§5.1)
# --------------------------------------------------------------------------- #
class InjectionAttemptError(ValueError):
    """Raised when a write-bound payload contains a prompt-injection / escape pattern.

    The orchestrator catches this and converts it into a recoverable error Observation —
    the write is aborted, the transaction cancelled, and the loop continues safely.
    """


# Maximum accepted length for any single string parameter destined for a write action.
# Legitimate primitives (strategy names, supplier ids) are short; long blobs are suspect.
MAX_WRITE_STRING_LEN = 100

# Patterns that should NEVER appear in a legitimate write parameter. These target common
# prompt-injection / command-injection / escape techniques rather than business values.
_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(?:the\s+)?(?:system|previous)", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"</?\s*(?:system|assistant|user)\s*>", re.IGNORECASE),  # role-tag injection
    re.compile(r"[;&|`$]{1,}\s*\w"),                                    # shell metacharacters
    re.compile(r"\$\([^)]*\)"),                                         # $(...) command subst
    re.compile(r"\\x[0-9a-fA-F]{2}"),                                   # hex escape sequences
    re.compile(r"\bset\s+\w+\s*="),                                     # override directives
)


def _scan_string(value: str, field: str) -> None:
    """Raise InjectionAttemptError if `value` is over-length or matches an injection pattern."""
    if len(value) > MAX_WRITE_STRING_LEN:
        raise InjectionAttemptError(
            f"field '{field}' exceeds max length {MAX_WRITE_STRING_LEN} "
            f"({len(value)} chars) — write aborted"
        )
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(value):
            raise InjectionAttemptError(
                f"field '{field}' matched injection pattern /{pattern.pattern}/ — write aborted"
            )


def _sanitize_value(field: str, value: Any) -> None:
    """Recursively scan a single value of ANY shape for injection/escape patterns (§5.1).

    Strings are scanned directly; dicts recurse over their values (and stringified keys);
    lists/tuples recurse over their elements. Non-string scalars (numbers, bools, None) are
    inert and pass through untouched. Handling lists is critical: an attacker could smuggle a
    prompt-override as a list element (e.g. {"x": ["ignore all previous instructions"]}) — a
    string-and-dict-only scanner would silently miss it. This closes that bypass.
    """
    if isinstance(value, str):
        _scan_string(value, field)
    elif isinstance(value, dict):
        for k, v in cast(Dict[Any, Any], value).items():
            # A malicious KEY is as dangerous as a malicious value — scan both.
            _scan_string(str(k), f"{field}.<key>")
            _sanitize_value(f"{field}.{k}", v)
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(cast(Any, value)):
            _sanitize_value(f"{field}[{i}]", item)
    # else: inert scalar (int/float/bool/None) — nothing to scan.


def sanitize_write_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scan every value in a write-bound payload for injection/escape patterns and
    length-boundary violations BEFORE it is allowed to mutate the ledger (§5.1).

    Handles arbitrarily-nested structures via `_sanitize_value`: strings, dicts (values AND
    keys), and LISTS/tuples are all traversed; non-string scalars pass through untouched. On
    any violation an InjectionAttemptError is raised so the caller aborts the write and
    cancels the transaction.

    Args:
        payload: The dict of parameters destined for a write/commit action.

    Returns:
        The same payload unchanged if it is clean (so callers can inline the call).
    """
    for key, value in payload.items():
        _scan_string(str(key), "<key>")
        _sanitize_value(str(key), value)

    return payload



__all__ = [
    "SPEND_AUTHORITY_LIMIT_USD",
    "SpendAuthorityResult",
    "check_spend_authority",
    "InjectionAttemptError",
    "MAX_WRITE_STRING_LEN",
    "sanitize_write_payload",
]
