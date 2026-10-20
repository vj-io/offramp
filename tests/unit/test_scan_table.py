"""The `offramp scan` table: call shape per site and the list-price ratio to the cheapest Nemotron tier."""

from offramp import cli, scan_table
from offramp.agents.scanner import scan_repo

APP = """from anthropic import Anthropic
from openai import OpenAI

client = OpenAI()
claude = Anthropic()
ROUTER = "Classify the ticket into exactly one of: billing, shipping, other. Reply with the label only."


def route(ticket):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": ROUTER}, {"role": "user", "content": ticket}],
    )


def act(ticket):
    return claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=256,
        tools=[{"name": "refund", "input_schema": {"type": "object"}}],
        messages=[{"role": "user", "content": ticket}],
    )


def extract(email):
    return client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": email}],
    )


def ask_json(email):
    return client.chat.completions.create(
        model="o3",
        messages=[{"role": "user", "content": f"Return JSON with the order id in: {email}"}],
    )


def reply(ticket):
    return client.chat.completions.create(
        model="gpt-4.1-mini",
        stream=True,
        messages=[{"role": "user", "content": f"Write a friendly reply to: {ticket}"}],
    )


def chat(history, model):
    return client.chat.completions.create(model=model, messages=history)


def legacy(text):
    return client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": text}])


def embed(text):
    return client.embeddings.create(model="text-embedding-3-small", input=text)
"""


def rows(tmp_path):
    (tmp_path / "app.py").write_text(APP)
    return {r["id"]: r for r in scan_table.table(scan_repo(tmp_path))["rows"]}


def test_shape_comes_from_the_call_parameters_when_they_settle_it(tmp_path):
    r = rows(tmp_path)
    assert (r["cs_02"]["shape"], r["cs_02"]["shape_guessed"]) == ("tool call", False)
    assert (r["cs_03"]["shape"], r["cs_03"]["shape_guessed"]) == ("JSON", False)
    assert (r["emb_01"]["shape"], r["emb_01"]["shape_guessed"]) == ("embedding", False)


def test_shape_is_guessed_from_the_prompt_text_otherwise(tmp_path):
    r = rows(tmp_path)
    assert (r["cs_01"]["shape"], r["cs_01"]["shape_guessed"]) == ("classification", True)
    assert (r["cs_04"]["shape"], r["cs_04"]["shape_guessed"]) == ("JSON", True)
    assert (r["cs_05"]["shape"], r["cs_05"]["shape_guessed"]) == ("free text", True)
    assert (r["cs_06"]["shape"], r["cs_06"]["shape_guessed"]) == ("unknown", False)  # prompt built at runtime


def test_streaming_and_runtime_prompts_are_flagged(tmp_path):
    r = rows(tmp_path)
    assert r["cs_05"]["streaming"] is True and "streams" in r["cs_05"]["flags"]
    assert r["cs_01"]["streaming"] is False and r["cs_01"]["flags"] == []
    assert "prompt built at runtime" in r["cs_06"]["flags"]


def test_list_price_ratio_against_each_nemotron_tier(tmp_path):
    r = rows(tmp_path)
    # pricing.yaml per 1M tokens: gpt-4o-mini 0.15/0.60, claude-sonnet-4-5 3.00/15.00;
    # Lightning 0.06/0.24, Super 0.30/0.90, Ultra 1.00/3.00
    assert r["cs_01"]["price_ratio"] == {
        "lightning": {"input": 2.5, "output": 2.5},
        "super": {"input": 0.5, "output": 0.67},  # below 1: the source model lists cheaper than Super
        "ultra": {"input": 0.15, "output": 0.2},
    }
    assert r["cs_02"]["price_ratio"] == {
        "lightning": {"input": 50.0, "output": 62.5},
        "super": {"input": 10.0, "output": 16.67},
        "ultra": {"input": 3.0, "output": 5.0},
    }
    assert r["cs_06"]["price_ratio"] is None and r["cs_06"]["price_note"] == "model set at runtime"
    assert r["cs_07"]["price_ratio"] is None and r["cs_07"]["price_note"] == "no list price on file"
    assert r["emb_01"]["price_ratio"] is None and r["emb_01"]["price_note"] == "no Nemotron embedding tier"
    refs = scan_table.references()
    assert [(ref["tier"], ref["model"]) for ref in refs] == [
        ("lightning", "nvidia/Nemotron-3_5-Lightning"),
        ("super", "nvidia/nemotron-3-super-120b-a12b"),
        ("ultra", "nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]


def test_rendered_table_labels_the_ratio_as_list_price_not_parity(tmp_path):
    (tmp_path / "app.py").write_text(APP)
    text = scan_table.render(scan_repo(tmp_path))
    assert "8 call sites" in text
    assert "list-price ratio, not measured parity" in text
    lines = text.splitlines()
    assert "list-price ratio vs Lightning / Super / Ultra*" in lines[2]
    row = next(line for line in lines if line.startswith("cs_02"))
    assert "app.py:17" in row and "anthropic claude-sonnet-4-5" in row and "tool call" in row
    assert row.endswith("50–62.5× / 10–16.7× / 3–5×")  # input–output ratio per tier
    assert next(line for line in lines if line.startswith("cs_01")).endswith("2.5× / 0.5–0.67× / 0.15–0.2×")


def test_cli_prints_the_table_by_default_and_the_scan_json_with_json(tmp_path, capsys):
    (tmp_path / "app.py").write_text(APP)
    assert cli.main(["scan", str(tmp_path)]) == 0
    assert "list-price ratio, not measured parity" in capsys.readouterr().out
    assert cli.main(["scan", str(tmp_path), "--json"]) == 0
    assert '"sites"' in capsys.readouterr().out


def test_no_sites_says_so(tmp_path):
    (tmp_path / "app.py").write_text("print('hello')\n")
    assert "No OpenAI, Anthropic, LiteLLM or LangChain call sites found" in scan_table.render(
        scan_repo(tmp_path)
    )


def test_non_text_system_parameter_does_not_break_the_shape(tmp_path):
    # Anthropic system blocks (a list) with messages built at runtime: no prompt text to read
    (tmp_path / "app.py").write_text(
        "from anthropic import Anthropic\n\nclaude = Anthropic()\n\n\ndef ask(history):\n"
        "    return claude.messages.create(\n"
        '        model="claude-haiku-4-5", max_tokens=64, messages=history,\n'
        '        system=[{"type": "text", "text": "Classify the ticket."}],\n'
        "    )\n"
    )
    [row] = scan_table.table(scan_repo(tmp_path))["rows"]
    assert (row["shape"], row["shape_guessed"]) == ("unknown", False)
