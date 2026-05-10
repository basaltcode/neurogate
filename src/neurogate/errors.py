from __future__ import annotations

from enum import Enum

import httpx


class ErrorCategory(str, Enum):
    """Why a provider call did/didn't complete. Drives both metrics labels and routing."""

    SUCCESS = "success"
    RATE_LIMIT = "rate_limit"          # 429, quota, resource_exhausted, overloaded
    SERVER_ERROR = "server_error"      # 5xx, network-layer, unavailable
    TIMEOUT = "timeout"                # httpx timeout or "timed out" text
    CONTEXT_EXCEEDED = "context_exceeded"  # 413, context/token length
    OUTPUT_LIMIT = "output_limit"      # requested max_tokens above provider's per-call output cap
    MODEL_DEAD = "model_dead"          # decommissioned / not found — config is stale
    EMPTY = "empty"                    # provider returned no text and no tool calls
    NO_TOOL_SUPPORT = "no_tool_support"  # pre-flight: tools requested, provider has none
    NO_REASONING_SUPPORT = "no_reasoning_support"  # pre-flight: reasoning requested, provider has none
    NO_VISION_SUPPORT = "no_vision_support"  # pre-flight: image content present, provider is text-only
    RATE_CAPPED = "rate_capped"        # pre-flight: local RPM/RPD cap reached
    OTHER = "other"                    # auth / bad request / unknown — non-retryable

    @property
    def retryable(self) -> bool:
        """If True, the router should try the next provider in the chain."""
        return self in _RETRYABLE

    @property
    def skips_smaller_context(self) -> bool:
        """If True, any downstream provider with a known-smaller context window
        should be skipped — whatever overflowed this call will overflow a smaller one."""
        return self is ErrorCategory.CONTEXT_EXCEEDED


_RETRYABLE = frozenset({
    ErrorCategory.RATE_LIMIT,
    ErrorCategory.SERVER_ERROR,
    ErrorCategory.TIMEOUT,
    ErrorCategory.CONTEXT_EXCEEDED,
    ErrorCategory.OUTPUT_LIMIT,
    ErrorCategory.MODEL_DEAD,
    ErrorCategory.EMPTY,
})


def classify(exc: BaseException) -> ErrorCategory:
    """Map a raised exception to its category. Order matters — check specific before generic."""
    if isinstance(exc, httpx.TimeoutException):
        return ErrorCategory.TIMEOUT
    if isinstance(exc, (httpx.NetworkError, httpx.RemoteProtocolError)):
        return ErrorCategory.SERVER_ERROR

    msg = str(exc).lower()

    if any(t in msg for t in ("429", "resource_exhausted", "quota", "rate limit")):
        return ErrorCategory.RATE_LIMIT
    if any(t in msg for t in ("decommissioned", "model_not_found", "model_decommissioned")):
        return ErrorCategory.MODEL_DEAD
    # Provider-side cap on output tokens (distinct from context-exceeded: input fits,
    # but the requested max_tokens is above the model's per-call output limit).
    # Groq qwen3-32b: "max_tokens must be less than or equal to 40960".
    if ("max_tokens" in msg or "max_output_tokens" in msg) and any(t in msg for t in (
        "must be less", "must be ≤", "cannot exceed", "exceeds the maximum",
        "maximum allowed", "maximum value",
    )):
        return ErrorCategory.OUTPUT_LIMIT
    # 413 is technically 4xx — check before generic 5xx group.
    if any(t in msg for t in (
        "413",
        "context length", "context_length", "maximum context",
        "tokens_limit", "too large", "too long", "reduce the length",
    )):
        return ErrorCategory.CONTEXT_EXCEEDED
    # Cloudflare 52x (520/522/523/524) fronts many providers (OpenRouter, Groq edges) —
    # they're transient gateway failures, must fall through to next provider, not abort chain.
    if any(t in msg for t in (
        "500", "502", "503", "504",
        "520", "522", "523", "524", "525", "526", "527", "529",
        "unavailable", "overloaded",
    )):
        return ErrorCategory.SERVER_ERROR
    if any(t in msg for t in ("timeout", "timed out")):
        return ErrorCategory.TIMEOUT
    if "empty response" in msg:
        return ErrorCategory.EMPTY
    return ErrorCategory.OTHER
