"""Per-run cost guardrails (PRD §10, Global Constraint 8). Only uncached calls are ever charged."""

import threading
from typing import Any

from offramp.schemas import Tier

# Eval-target calls (role starts with this) count toward the token caps but not the Ultra-agent cap.
# Every other Ultra call (case_gen adversarial, adapter, iterator, judge) is an Ultra *agent* call.
EVAL_ROLE_PREFIX = "eval"


class BudgetExceeded(Exception):
    """A per-run cap was reached; the pipeline turns this into KEEP_CLOSED(reason="budget ...")."""


class Budget:
    def __init__(self, max_in: int = 1_500_000, max_out: int = 300_000, max_ultra_agent_calls: int = 40):
        self.max_in, self.max_out, self.max_ultra_agent_calls = max_in, max_out, max_ultra_agent_calls
        self.tokens_in = self.tokens_out = self.ultra_agent_calls = 0
        self.spend_usd = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def _is_ultra_agent(role: str, tier: Tier) -> bool:
        return tier == Tier.ULTRA and not role.startswith(EVAL_ROLE_PREFIX)

    def check(self, role: str, tier: Tier) -> None:
        """Pre-call gate: refuse a call that cannot fit. Costs nothing, records nothing."""
        with self._lock:
            if self.tokens_in >= self.max_in or self.tokens_out >= self.max_out:
                raise BudgetExceeded(f"token cap reached ({self.tokens_in}/{self.max_in} in, "
                                     f"{self.tokens_out}/{self.max_out} out)")
            if self._is_ultra_agent(role, tier) and self.ultra_agent_calls >= self.max_ultra_agent_calls:
                raise BudgetExceeded(f"Ultra agent call cap reached ({self.max_ultra_agent_calls})")

    def charge(self, role: str, tier: Tier, tin: int, tout: int, cost_usd: float = 0.0) -> None:
        """Record a completed uncached call; raise if it pushed a cap over.

        The spend is real, so it is recorded before raising. Concurrent callers that all passed check()
        can overshoot a cap by at most the number of in-flight calls.
        """
        with self._lock:
            self.tokens_in += tin
            self.tokens_out += tout
            self.spend_usd += cost_usd
            if self._is_ultra_agent(role, tier):
                self.ultra_agent_calls += 1
            if self.tokens_in > self.max_in or self.tokens_out > self.max_out:
                raise BudgetExceeded(f"token cap exceeded ({self.tokens_in}/{self.max_in} in, "
                                     f"{self.tokens_out}/{self.max_out} out)")
            if self.ultra_agent_calls > self.max_ultra_agent_calls:
                raise BudgetExceeded(f"Ultra agent call cap exceeded ({self.max_ultra_agent_calls})")

    def snapshot(self) -> dict[str, Any]:
        """Key names match RunTotals where they overlap."""
        with self._lock:
            return {
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "ultra_agent_calls": self.ultra_agent_calls,
                "tf_spend_usd": self.spend_usd,
                "max_in": self.max_in,
                "max_out": self.max_out,
                "max_ultra_agent_calls": self.max_ultra_agent_calls,
            }
