"""Reference implementation of shop/refund.py's CHARGE_PROMPT (test fixture oracle)."""

import random
from decimal import ROUND_HALF_UP, Decimal

SITE_FILE = "shop/refund.py"


def answer(params: dict) -> str:
    total = params["seats"] * Decimal(params["price"]) * (1 - Decimal(params["coupon"]) / 100)
    return str(total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def sample(seed: int, n: int, hard: bool = False) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        coupon = rng.choice(["100", "33", "12.5"]) if hard else rng.choice(["0", "10", "25"])
        tags = ["hard"] if hard else []
        out.append({"id": f"s-{i}", "tags": tags, "seats": rng.randint(1, 40),
                    "price": f"{rng.randint(100, 9999) / 100:.2f}", "coupon": coupon})
    return out
