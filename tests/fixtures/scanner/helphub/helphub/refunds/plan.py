"""Work out how much to refund when a customer cancels a subscription mid-period."""

import re
from datetime import date
from decimal import Decimal

from openai import OpenAI
from pydantic import BaseModel

REFUND_POLICY = """You compute subscription refunds for HelpHub billing. Follow this policy \
exactly, step by step, and show your arithmetic.

1. Tiered price. The list price is in USD per seat per billing period. Seats are priced in \
graduated tiers: seats 1-10 cost 100% of list price, seats 11-50 cost 85%, seats 51 and above \
cost 70%. Subtotal = sum over the tiers of (seats in that tier x list price x tier rate). \
Round the subtotal to 2 decimals, half up.
2. Coupon. Take the coupon percentage off the subtotal and round to 2 decimals, half up. The \
result is the USD amount paid for the period.
3. Days used. Days used = the number of days from the period start date to the cancellation \
date, counting both dates. Unused days = billing period days - days used, never below 0.
4. Refundable amount in USD. If the billing period is 365 days and days used is 14 or fewer, \
the refundable amount is the full amount paid. Otherwise, prorated = amount paid x unused \
days / billing period days, rounded to 2 decimals half up; then subtract a 3% processing fee \
on the prorated amount, rounded to 2 decimals half up. The fee is not charged on a full refund.
5. Prior refunds. Subtract the USD refunds already issued from the refundable amount. If the \
result is negative, it is 0.
6. Currency. Multiply by the exchange rate recorded at purchase (units of the customer's \
currency per 1 USD) and round half up to the currency's minor unit: 2 decimals for USD, EUR \
and GBP, 0 decimals for JPY.
7. Show your working, then put the final refund in the customer's currency alone on the last \
line as `ANSWER: <number>` (digits and a decimal point only: no currency symbol, no thousands \
separator)."""


class RefundRequest(BaseModel):
    plan_name: str
    period_days: int  # length of the billing period: 30 (monthly) or 365 (annual)
    list_price_usd: Decimal  # per seat per period
    seats: int
    period_start: date
    cancel_date: date
    coupon_pct: Decimal
    prior_refund_usd: Decimal
    currency: str  # what the customer paid in: USD, EUR, GBP or JPY
    fx_rate: Decimal  # units of `currency` per 1 USD, recorded at purchase


class RefundPlan(BaseModel):
    amount: Decimal  # refund in the customer's currency
    working: str  # the model's full reply


class PlanParseError(ValueError):
    """The reply did not end with an `ANSWER: <number>` line."""


ANSWER_LINE = re.compile(r"ANSWER:\s*\**\s*(-?[\d,]*\.?\d+)")


def parse_answer(text: str) -> Decimal:
    matches = ANSWER_LINE.findall(text)
    if not matches:
        raise PlanParseError(f"no ANSWER line in model reply: {text[-80:]!r}")
    return Decimal(matches[-1].replace(",", ""))


def plan_refund(client: OpenAI, request: RefundRequest) -> RefundPlan:
    response = client.chat.completions.create(
        model="o3",
        max_completion_tokens=16000,
        messages=[
            {"role": "system", "content": REFUND_POLICY},
            {
                "role": "user",
                "content": (
                    f"Plan: {request.plan_name}, billing period {request.period_days} days\n"
                    f"List price: {request.list_price_usd} USD per seat per period\n"
                    f"Seats: {request.seats}\n"
                    f"Period start: {request.period_start}\n"
                    f"Cancelled on: {request.cancel_date}\n"
                    f"Coupon: {request.coupon_pct}% off\n"
                    f"Refunds already issued: {request.prior_refund_usd} USD\n"
                    f"Customer paid in {request.currency} at {request.fx_rate} "
                    f"{request.currency} per 1 USD\n\n"
                    f"How much do we refund, in {request.currency}?"
                ),
            },
        ],
    )
    reply = response.choices[0].message.content or ""
    return RefundPlan(amount=parse_answer(reply), working=reply)
