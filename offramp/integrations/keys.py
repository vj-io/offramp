"""Primary -> fallback provider keys (Global Constraint 28).

Every provider key NAME has an optional NAME_FALLBACK twin. When the primary is rejected or out of credit the
process moves to the fallback for good and the caller retries the same request once. Keys are never logged,
cached or put in an exception message: this module only ever compares them.
"""

import logging
import os
import re
from typing import Any

log = logging.getLogger(__name__)

# 432/433 are Tavily's own "plan / pay-as-you-go limit exceeded" statuses (out of credit, not a rate limit).
_REJECTED = {401, 402, 403, 432, 433}
# A 429 whose body says this is out of credit, not a rate limit.
_OUT_OF_CREDIT = re.compile(r"quota|credit|balance|insufficient", re.I)

_exhausted: set[str] = set()  # providers whose primary key was rejected; process-wide on purpose


def current(provider: str, primary: str | None) -> str | None:
    """The key to use now: the provider's `<PROVIDER>_API_KEY_FALLBACK` once its primary has been rejected."""
    fallback = os.environ.get(f"{provider.upper()}_API_KEY_FALLBACK")
    return fallback if fallback and provider in _exhausted else primary


def switch_to_fallback(provider: str, used_key: str | None, status: int | None, body: str = "") -> bool:
    """True when this rejection means the caller should retry once with the fallback key.

    `used_key` is the key the failed request carried: a rejected fallback is never retried, and concurrent
    callers that all failed on the primary all get True (the switch is logged once).
    """
    fallback = os.environ.get(f"{provider.upper()}_API_KEY_FALLBACK")
    if not fallback or used_key == fallback:
        return False
    if status not in _REJECTED and not (status == 429 and _OUT_OF_CREDIT.search(body)):
        return False
    if provider not in _exhausted:
        _exhausted.add(provider)
        log.warning("%s: primary key exhausted, using fallback", provider)
    return True


def status_and_body(exc: BaseException) -> tuple[int | None, str]:
    """HTTP status and body of the response behind an SDK exception (openai, requests, httpx), if any."""
    seen: BaseException | None = exc
    while seen is not None:
        response: Any = getattr(seen, "response", None)
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status, str(getattr(response, "text", ""))[:2000]
        seen = seen.__cause__ or seen.__context__
    return None, ""
