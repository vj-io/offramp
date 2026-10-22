"""Review R1-1: a module client whose name is also a function's parameter is never read in its module, so
this case lives apart from app.py's (there, it made the module client unreadable and their checks vacuous)."""

from openai import OpenAI

GIVEN = OpenAI(timeout=5)


def summarise_with_given_client(GIVEN, text: str) -> str:  # a parameter shadows the module client
    resp = GIVEN.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": text}])
    return resp.choices[0].message.content


def module_client_also_a_parameter(text: str) -> str:  # the module name is a parameter elsewhere: unsure
    resp = GIVEN.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": text}])
    return resp.choices[0].message.content
