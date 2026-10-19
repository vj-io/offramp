"""Errors that end an evaluation without producing a measured result.

A ProviderError says the provider refused or could not be reached (bad key, no credit, quota, unknown model,
still rate-limited after the SDK's retries, gateway/service outage 502/503/504, network down), or that the
judge could not judge. It says nothing about the model under test, so it is a run error the pipeline reports,
never a failed case: scoring it would show "0% parity" that was never measured.
The same holds when both keys are exhausted after the GC28 fallback switch: the client re-raises that error.
"""

import openai


class ProviderError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status  # HTTP status, None for a network error


class MissingReference(ValueError):
    """Deterministic-site cases of the split have no reference to be scored against (a data error)."""

    def __init__(self, case_ids: list[str]):
        super().__init__(f"{len(case_ids)} case(s) have no reference: {', '.join(case_ids)}")
        self.case_ids = case_ids


OUTAGE_STATUSES = (502, 503, 504)


def is_case_failure(e: openai.APIError) -> bool:
    """A per-request outcome of the model service, scored as a failed case: a timeout (a model can be too
    slow), a 500 after the SDK's retries, a 400/422 (e.g. the input does not fit the model's context), or a
    response the SDK cannot parse. Everything else, 502/503/504 included, is a ProviderError."""
    if isinstance(e, openai.APITimeoutError | openai.APIResponseValidationError):
        return True
    if not isinstance(e, openai.APIStatusError):
        return False
    return e.status_code in (400, 422) or (e.status_code >= 500 and e.status_code not in OUTAGE_STATUSES)


def is_timeout_or_5xx(e: openai.APIError) -> bool:
    return isinstance(e, openai.APITimeoutError) or (
        isinstance(e, openai.APIStatusError) and e.status_code >= 500
    )


def provider_error(e: openai.APIError, who: str = "") -> ProviderError:
    status = getattr(e, "status_code", None)
    where = f" (HTTP {status})" if status else ""
    return ProviderError(f"{who}{type(e).__name__}{where}: {e}"[:500], status)
