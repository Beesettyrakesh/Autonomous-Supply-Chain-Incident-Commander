"""
Deterministic, code-enforced safety barriers.

The LLM has no control over these; they intercept the execution path after the State
Mutation Layer commits a change. The module is dependency-light (pure functions + small
dataclasses, no orchestrator imports) so each barrier is independently testable and reusable
by the CLI and dashboard.

Barriers:
  1. Financial Spend-Authority Guardrail — spend <= limit may be auto-approved; spend above
     it pauses for a human-in-the-loop (HITL) approve/reject decision.
  2. Jailbreak / Injection Sanitization — a regex + length-boundary scan applied to any
     parameter destined for a write action; a hit aborts the write.

The spend-authority limit defaults to $20,000 and is overridable via the
`SPEND_AUTHORITY_LIMIT_USD` environment variable.
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
    """Read the delegated spend-authority limit (USD); fixed policy, env-overridable."""
    raw = os.environ.get("SPEND_AUTHORITY_LIMIT_USD", "20000")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 20000.0


# Delegated approval authority: spend at or below this may be auto-approved; above it is
# escalated to a human (HITL).
SPEND_AUTHORITY_LIMIT_USD: float = _load_spend_authority_limit()


# --------------------------------------------------------------------------- #
# 1. Financial Spend-Authority Guardrail
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpendAuthorityResult:
    """Outcome of a spend-authority check — flat, ledger-safe primitives only."""

    within_authority: bool          # True -> auto-approve; False -> needs HITL
    spend_usd: float                # evaluated total spend (unit price * qty)
    limit_usd: float                # delegated authority limit checked against
    supplier_id: str                # the vendor the spend is with
    reason: str                     # human-readable explanation for the record


def check_spend_authority(
    supplier_id: str,
    unit_price_usd: float,
    quantity: int,
    limit_usd: float = SPEND_AUTHORITY_LIMIT_USD,
) -> SpendAuthorityResult:
    """
    Evaluate whether a negotiated purchase is within the agent's delegated spend authority.

    Wraps `decision_helpers.policy_check` (approved-vendor + spend-cap compliance) as the
    single source of truth for the rule, and enriches the boolean into a structured result.

    Args:
        supplier_id: The winning vendor the PO would be placed with.
        unit_price_usd: The negotiated per-unit price.
        quantity: The order quantity.
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
# 2. Jailbreak / Injection Sanitization
# --------------------------------------------------------------------------- #
class InjectionAttemptError(ValueError):
    """Raised when a write-bound payload contains a prompt-injection / escape pattern.

    The orchestrator catches this and converts it into a recoverable error Observation: the
    write is aborted, the transaction cancelled, and the loop continues safely.
    """


# Max length for a system-identifier string destined for a write action (strategy names,
# supplier ids, tool args). These are short by design, so a long blob here is suspect.
MAX_WRITE_STRING_LEN = 100

# Looser bound for legitimate natural-language fields (e.g. a vendor's raw reply), which can
# exceed 100 chars. Callers pass this via `max_len` only for genuine NL text; the injection
# regex still applies, so injection payloads are blocked regardless of length.
MAX_NL_STRING_LEN = 500


# Patterns that should never appear in a legitimate write parameter — common
# prompt-injection / command-injection / escape techniques, not business values.
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


def _scan_string(value: str, field: str, max_len: int = MAX_WRITE_STRING_LEN) -> None:
    """Raise InjectionAttemptError if `value` is over-length or matches an injection pattern.

    The injection regex scan always runs regardless of the length bound, so a looser NL
    length allowance never weakens content detection.
    """
    if len(value) > max_len:
        raise InjectionAttemptError(
            f"field '{field}' exceeds max length {max_len} "
            f"({len(value)} chars) — write aborted"
        )
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(value):
            raise InjectionAttemptError(
                f"field '{field}' matched injection pattern /{pattern.pattern}/ — write aborted"
            )


def _sanitize_value(field: str, value: Any, max_len: int = MAX_WRITE_STRING_LEN) -> None:
    """Recursively scan a value of any shape for injection/escape patterns.

    Strings are scanned directly; dicts recurse over values and stringified keys; lists and
    tuples recurse over elements (closing the bypass of smuggling an override as a list
    element). Non-string scalars are inert. Dict keys always use the strict identifier bound.
    """
    if isinstance(value, str):
        _scan_string(value, field, max_len)
    elif isinstance(value, dict):
        for k, v in cast(Dict[Any, Any], value).items():
            _scan_string(str(k), f"{field}.<key>")
            _sanitize_value(f"{field}.{k}", v, max_len)
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(cast(Any, value)):
            _sanitize_value(f"{field}[{i}]", item, max_len)


def sanitize_write_payload(
    payload: Dict[str, Any], max_len: int = MAX_WRITE_STRING_LEN
) -> Dict[str, Any]:
    """
    Scan every value in a write-bound payload for injection/escape patterns and length
    violations before it is allowed to mutate the ledger.

    Handles arbitrarily-nested structures (strings, dict values and keys, lists/tuples). On
    any violation an InjectionAttemptError is raised so the caller aborts the write.

    Args:
        payload: The dict of parameters destined for a write/commit action.
        max_len: Per-string length bound. Defaults to the strict identifier bound; pass
            MAX_NL_STRING_LEN for genuine natural-language fields.

    Returns:
        The same payload unchanged if it is clean (so callers can inline the call).
    """
    for key, value in payload.items():
        _scan_string(str(key), "<key>")
        _sanitize_value(str(key), value, max_len)

    return payload


__all__ = [
    "SPEND_AUTHORITY_LIMIT_USD",
    "SpendAuthorityResult",
    "check_spend_authority",
    "InjectionAttemptError",
    "MAX_WRITE_STRING_LEN",
    "MAX_NL_STRING_LEN",
    "sanitize_write_payload",
]
