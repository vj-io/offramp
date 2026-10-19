"""Token Factory access layer: client, cache, tiers, budget (see docs/PRD.md §6.11, §10)."""

from offramp.tf.budget import Budget, BudgetExceeded
from offramp.tf.cache import Cache, cache_key
from offramp.tf.client import LLM, CacheMiss, ChatResult, TFClient
from offramp.tf.fake import FakeLLM
from offramp.tf.tiers import TierInfo, get_tier, reasoning_params

__all__ = [
    "LLM",
    "Budget",
    "BudgetExceeded",
    "Cache",
    "CacheMiss",
    "ChatResult",
    "FakeLLM",
    "TFClient",
    "TierInfo",
    "cache_key",
    "get_tier",
    "reasoning_params",
]
