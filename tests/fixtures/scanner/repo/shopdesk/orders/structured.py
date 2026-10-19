"""Parse a shipping address (fixture: AsyncOpenAI, response_format json_schema from a pydantic model)."""

from openai import AsyncOpenAI
from pydantic import BaseModel


class Address(BaseModel):
    street: str
    city: str
    postcode: str
    country: str


ADDRESS_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "address", "schema": Address.model_json_schema(), "strict": True},
}
aclient = AsyncOpenAI()


async def parse_address(text: str) -> Address:
    resp = await aclient.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Parse the shipping address:\n" + text}],
        response_format=ADDRESS_FORMAT,
    )
    return Address.model_validate_json(resp.choices[0].message.content)
