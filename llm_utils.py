"""
Shared resilience utilities for Google Gen AI calls.

Both the Incident Commander (`orchestrator.py`) and the Supplier-Persona sub-graph
(`negotiation_agent.py`) route their `generate_content` calls through `generate_with_retry`
so they obey one transient-error policy:

- Bounded retry/backoff on 429 (rate/quota) and 5xx (transient server) errors, plus httpx
  transport errors (dropped connection / timeout).
- Honors the server's suggested `RetryInfo.retryDelay` when present (capped).
- Per-DAY quota exhaustion is not retried — it raises `LLMUnavailableError` so callers
  escalate cleanly.
- Uses `asyncio.sleep` so concurrent negotiations don't block the event loop.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

LLM_MAX_RETRIES = 3
LLM_RETRY_CAP_SECONDS = 40.0  # never sleep longer than this on a single backoff


class LLMUnavailableError(RuntimeError):
    """Raised when the reasoning core is unreachable after exhausting retries.

    Callers catch this and escalate to HUMAN TAKEOVER (or a clean negotiation FAILED)
    rather than leaking a raw SDK traceback.
    """


def _suggested_retry_delay(exc: Any, attempt: int) -> float:
    """Extract the server's RetryInfo.retryDelay (e.g. '34s') if present; else backoff."""
    try:
        details = exc.details.get("error", {}).get("details", [])  # type: ignore[attr-defined]
        for d in details:
            if d.get("@type", "").endswith("RetryInfo"):
                raw = str(d.get("retryDelay", "")).rstrip("s")
                return float(raw)
    except Exception:  # pragma: no cover - fall back to exponential backoff
        pass
    return 2.0 ** attempt


def _is_daily_quota_error(exc: Any) -> bool:
    """True if a 429 is a per-DAY quota exhaustion (unrecoverable by waiting).

    Per-day quotas carry a quotaId like 'GenerateRequestsPerDayPerProjectPerModel'; per-minute
    quotas can recover with a short backoff, so only the daily kind skips retries. Best-effort
    against QuotaFailure details, with a substring fallback on the raw message.
    """
    try:
        details = exc.details.get("error", {}).get("details", [])  # type: ignore[attr-defined]
        for d in details:
            if d.get("@type", "").endswith("QuotaFailure"):
                for v in d.get("violations", []):
                    if "PerDay" in str(v.get("quotaId", "")):
                        return True
    except Exception:  # pragma: no cover - fall back to message scan
        pass
    return "PerDay" in str(exc)


async def generate_with_retry(
    client: Any,
    *,
    model: str,
    contents: Any,
    config: Any,
    source: str = "llm",
    log: Optional[Any] = None,
) -> Any:
    """Call `client.aio.models.generate_content` with bounded retry/backoff.

    Args:
        client: An initialized google-genai Client (must expose `.aio.models`).
        model: Model id (e.g. "gemini-2.5-flash").
        contents: Prompt contents passed to the SDK.
        config: A `types.GenerateContentConfig` instance.
        source: Label used in log lines (e.g. "orchestrator" / "negotiation:SUP-C").
        log: Optional callable `log(source, message)` for audit logging. If None, retries
            happen silently.

    Returns:
        The SDK response object.

    Raises:
        LLMUnavailableError: on daily-quota exhaustion, non-retryable client errors, or after
            exhausting `LLM_MAX_RETRIES` on transient 429/5xx/transport errors.
    """
    from google.genai import errors as genai_errors  # type: ignore

    # httpx is google-genai's HTTP transport, so it is importable here. Its transport-level
    # errors (dropped connection / timeout) are transient and usually succeed on retry, so we
    # treat them like a 5xx rather than let them crash the ReAct loop.
    transport_errors: tuple[type[BaseException], ...]
    try:
        import httpx  # type: ignore

        transport_errors = (httpx.TransportError,)
    except Exception:  # pragma: no cover - httpx should always be present
        transport_errors = ()

    def _log(msg: str) -> None:
        if log is not None:
            try:
                log(source, msg)
            except Exception:  # pragma: no cover - logging must never crash the call
                pass

    last_exc: Optional[Exception] = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            return await client.aio.models.generate_content(
                model=model, contents=contents, config=config
            )
        except genai_errors.ClientError as exc:
            last_exc = exc
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            # Only 429 is worth retrying among client errors.
            if status != 429:
                raise LLMUnavailableError(f"LLM client error {status}: {exc}") from exc
            # Daily-quota 429s cannot recover by waiting — escalate immediately.
            if _is_daily_quota_error(exc):
                _log("LLM 429 DAILY-quota exhausted; not retrying")
                raise LLMUnavailableError(
                    f"LLM daily quota exhausted (no retry): {exc}"
                ) from exc
            delay = min(_suggested_retry_delay(exc, attempt), LLM_RETRY_CAP_SECONDS)
            _log(f"LLM 429 (per-minute) attempt={attempt}/{LLM_MAX_RETRIES}; backing off {delay:.1f}s")
            if attempt < LLM_MAX_RETRIES:
                await asyncio.sleep(delay)
        except genai_errors.ServerError as exc:  # 5xx — transient server-side
            last_exc = exc
            delay = min(2.0 ** attempt, LLM_RETRY_CAP_SECONDS)
            _log(f"LLM 5xx attempt={attempt}/{LLM_MAX_RETRIES}; backing off {delay:.1f}s")
            if attempt < LLM_MAX_RETRIES:
                await asyncio.sleep(delay)
        except transport_errors as exc:  # dropped connection / read-write / timeout
            last_exc = exc
            delay = min(2.0 ** attempt, LLM_RETRY_CAP_SECONDS)
            _log(
                f"LLM transport error ({type(exc).__name__}) attempt={attempt}/{LLM_MAX_RETRIES}; "
                f"backing off {delay:.1f}s"
            )
            if attempt < LLM_MAX_RETRIES:
                await asyncio.sleep(delay)

    raise LLMUnavailableError(
        f"LLM unavailable after {LLM_MAX_RETRIES} attempts: {last_exc}"
    )


__all__ = [
    "LLMUnavailableError",
    "LLM_MAX_RETRIES",
    "LLM_RETRY_CAP_SECONDS",
    "generate_with_retry",
]
