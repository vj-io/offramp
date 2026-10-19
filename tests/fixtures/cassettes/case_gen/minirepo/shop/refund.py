"""Compute what to charge for seats after a coupon."""

CHARGE_PROMPT = """Compute the charge: seats x price per seat, then take the coupon percentage off.
Round to 2 decimals, half up. Show your working, then put the result alone on the last line as
`ANSWER: <number>`."""


def charge(client, req):
    return client.chat.completions.create(
        model="o3",
        messages=[
            {"role": "system", "content": CHARGE_PROMPT},
            {
                "role": "user",
                "content": f"Seats: {req.seats}\nPrice per seat: {req.price} USD\nCoupon: {req.coupon}% off",
            },
        ],
    )
