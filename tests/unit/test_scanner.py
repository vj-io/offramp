from pathlib import Path

import pytest

from offramp.agents.scanner import scan_repo
from offramp.schemas import CallSite, PromptTemplate

REPO = Path(__file__).parents[1] / "fixtures" / "scanner" / "repo"
NRC = "NEEDS_RUNTIME_CAPTURE"

AGENT_SYSTEM = "You are a support operations agent. Call exactly one tool."
CLASSIFY_SYSTEM = (
    "Classify the support ticket into exactly one queue: billing, shipping, returns, technical, account, "
    "other. Reply with the queue name only."
)
EXTRACT_SYSTEM = (
    "Extract the order fields from the email. Reply with JSON matching this schema:\n<pydantic:OrderFields>"
)
PLAN_PROMPT = (
    "Compute the refund for this subscription change.\nPlan: {plan}\nDays used this cycle: {days_used}\n"
    "Show your working, then finish with a final line `ANSWER: <number>`."
)
QA_SYSTEM = "Answer using only the numbered knowledge-base snippets and cite them like [2]."
REPLY_INSTRUCTIONS = "Write a short, friendly reply to the customer. Never promise refunds."
SUMMARY_SYSTEM = "Summarise the ticket thread for the next agent in at most three bullet points."


def tpl(user: str, system: str | None = None, *variables: str) -> PromptTemplate:
    return PromptTemplate(system=system, user=user, variables=list(variables))


EXPECTED = [
    CallSite(
        id="cs_01",
        file="shopdesk/assist/dynamic.py",
        lines=[13, 13],
        sdk="openai",
        call="client.chat.completions.create",
        function="continue_chat",
        source_model="gpt-4o-mini",
        prompt_source="dynamic",
        flags=[NRC],
    ),
    CallSite(
        id="cs_02",
        file="shopdesk/assist/reply.py",
        lines=[13, 18],
        sdk="openai",
        call="self.client.responses.create",
        function="draft",
        source_model="gpt-4.1-mini",
        prompt_source="fstring",
        prompt_ref="REPLY_INSTRUCTIONS",
        template=tpl(
            "Customer {customer_name} wrote:\n{message}", REPLY_INSTRUCTIONS, "customer_name", "message"
        ),
        params={"instructions": REPLY_INSTRUCTIONS, "max_output_tokens": 400},
    ),
    CallSite(
        id="cs_03",
        file="shopdesk/assist/stream_reply.py",
        lines=[10, 17],
        sdk="openai",
        call="client.chat.completions.create",
        function="stream_reply",
        source_model="gpt-4o",
        prompt_source="variable",
        prompt_ref="STREAM_SYSTEM",
        template=tpl("{question}", "You are a concise support assistant.", "question"),
        params={"stream": True},
        flags=["MANUAL_REVIEW"],
    ),
    CallSite(
        id="cs_04",
        file="shopdesk/inbox/classify.py",
        lines=[14, 22],
        sdk="openai",
        call="client.chat.completions.create",
        function="classify_ticket",
        source_model="gpt-4o-mini",
        prompt_source="fstring",
        prompt_ref="CLASSIFY_SYSTEM",
        template=tpl("Subject: {subject}\n\n{body}", CLASSIFY_SYSTEM, "subject", "body"),
        params={"temperature": 0, "max_tokens": 5},
    ),
    CallSite(
        id="cs_05",
        file="shopdesk/legacy/lc_chain.py",
        lines=[6, 6],
        sdk="langchain",
        call="ChatOpenAI",
        source_model="gpt-4o",
        prompt_source="dynamic",
        params={"temperature": 0},
        flags=[NRC],
    ),
    CallSite(
        id="cs_06",
        file="shopdesk/legacy/lc_chain.py",
        lines=[10, 10],
        sdk="langchain",
        call="ChatAnthropic",
        function="make_fast_llm",
        source_model="claude-3-5-haiku-20241022",
        prompt_source="dynamic",
        params={"max_tokens": "<expr>"},
        flags=[NRC],
    ),
    CallSite(
        id="cs_07",
        file="shopdesk/legacy/litellm_reply.py",
        lines=[10, 14],
        sdk="litellm",
        call="litellm.completion",
        function="polite_reply",
        source_model="claude-3-5-haiku-20241022",
        prompt_source="fstring",
        template=tpl("Rewrite this reply in a polite tone:\n{message}", None, "message"),
        params={"temperature": 0.3},
    ),
    CallSite(
        id="cs_08",
        file="shopdesk/legacy/litellm_reply.py",
        lines=[19, 22],
        sdk="litellm",
        call="llm_completion",
        function="translate",
        source_model="gpt-4o-mini",
        prompt_source="fstring",
        template=tpl("Translate to {language}:\n{message}", None, "language", "message"),
    ),
    CallSite(
        id="cs_09",
        file="shopdesk/ops/agent.py",
        lines=[29, 36],
        sdk="anthropic",
        call="self.client.messages.create",
        function="next_step",
        source_model="claude-sonnet-4-5",
        prompt_source="fstring",
        prompt_ref="AGENT_SYSTEM",
        template=tpl("Ticket:\n{ticket}\n\nHistory so far:\n{history}", AGENT_SYSTEM, "ticket", "history"),
        params={
            "max_tokens": 1024,
            "system": AGENT_SYSTEM,
            "tools": [
                {
                    "name": "lookup_order",
                    "description": "Fetch an order by id.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    },
                },
                {
                    "name": "issue_refund",
                    "description": "Refund part or all of an order.",
                    "input_schema": "<pydantic:RefundArgs>",
                },
            ],
            "tool_choice": {"type": "any"},
        },
    ),
    CallSite(
        id="cs_10",
        file="shopdesk/orders/extract.py",
        lines=[23, 28],
        sdk="anthropic",
        call="client.messages.create",
        function="extract_order",
        source_model="claude-sonnet-4-5",
        prompt_source="fstring",
        prompt_ref="EXTRACT_SYSTEM",
        template=tpl(
            'Email:\n{email}\n\nReturn JSON only, e.g. {{"order_id": "A-1"}}.', EXTRACT_SYSTEM, "email"
        ),
        params={"max_tokens": 800, "system": EXTRACT_SYSTEM},
    ),
    CallSite(
        id="cs_11",
        file="shopdesk/orders/structured.py",
        lines=[22, 26],
        sdk="openai",
        call="aclient.chat.completions.create",
        function="parse_address",
        source_model="gpt-4.1-mini",
        prompt_source="fstring",
        template=tpl("Parse the shipping address:\n{text}", None, "text"),
        params={
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "address", "schema": "<pydantic:Address>", "strict": True},
            }
        },
    ),
    # providers/router.py: `client` and `self.client` each bound to both SDKs, in different scopes
    CallSite(
        id="cs_12",
        file="shopdesk/providers/router.py",
        lines=[9, 9],
        sdk="openai",
        call="client.chat.completions.create",
        function="ask_openai",
        source_model="gpt-4o-mini",
        prompt_source="variable",
        template=tpl("{question}", None, "question"),
    ),
    CallSite(
        id="cs_13",
        file="shopdesk/providers/router.py",
        lines=[15, 17],
        sdk="anthropic",
        call="client.messages.create",
        function="ask_claude",
        source_model="claude-haiku-4-5",
        prompt_source="variable",
        template=tpl("{question}", None, "question"),
        params={"max_tokens": 200},
    ),
    CallSite(
        id="cs_14",
        file="shopdesk/providers/router.py",
        lines=[26, 26],
        sdk="openai",
        call="self.client.chat.completions.create",
        function="complete",
        source_model="gpt-4o",
        prompt_source="variable",
        template=tpl("{prompt}", None, "prompt"),
    ),
    CallSite(
        id="cs_15",
        file="shopdesk/providers/router.py",
        lines=[35, 37],
        sdk="anthropic",
        call="self.client.messages.create",
        function="complete",
        source_model="claude-sonnet-4-5",
        prompt_source="variable",
        template=tpl("{prompt}", None, "prompt"),
        params={"max_tokens": 300},
    ),
    CallSite(
        id="cs_16",
        file="shopdesk/qa/answer.py",
        lines=[12, 19],
        sdk="openai",
        call="client.chat.completions.create",
        function="answer",
        source_model="gpt-4o",
        prompt_source="fstring",
        prompt_ref="QA_SYSTEM",
        template=tpl("Snippets:\n{context}\n\nQuestion: {question}", QA_SYSTEM, "context", "question"),
        params={"temperature": 0.2},
    ),
    CallSite(
        id="cs_17",
        file="shopdesk/refunds/plan.py",
        lines=[15, 20],
        sdk="openai",
        call="openai.chat.completions.create",
        function="plan_refund",
        source_model="o3",
        prompt_source="fstring",
        prompt_ref="PLAN_PROMPT",
        template=tpl(PLAN_PROMPT, None, "plan", "days_used"),
        params={"max_completion_tokens": 4000, "reasoning_effort": "high"},
    ),
    CallSite(
        id="cs_18",
        file="shopdesk/summaries/summarize.py",
        lines=[10, 15],
        sdk="anthropic",
        call="client.messages.create",
        function="summarize",
        source_model="claude-haiku-4-5",
        prompt_source="variable",
        prompt_ref="SUMMARY_SYSTEM",
        template=tpl("{thread}", SUMMARY_SYSTEM, "thread"),
        params={"max_tokens": 300, "system": [{"type": "text", "text": SUMMARY_SYSTEM}]},
    ),
    CallSite(
        id="emb_01",
        file="shopdesk/search/embed.py",
        lines=[10, 10],
        kind="embedding",
        sdk="openai",
        call="client.embeddings.create",
        function="embed",
        source_model="text-embedding-3-small",
        prompt_source="dynamic",
        flags=["EMBEDDINGS"],
    ),
]


@pytest.fixture(scope="module")
def scan():
    return scan_repo(REPO)


def test_exact_call_site_list(scan):
    assert (scan.repo, scan.warnings) == (str(REPO), [])
    assert scan.sites == EXPECTED


@pytest.mark.parametrize("expected", EXPECTED, ids=[e.id for e in EXPECTED])
def test_each_site(scan, expected):
    assert {s.id: s for s in scan.sites}[expected.id] == expected


def test_negatives_produce_no_sites(scan):
    # openai imported but unused, plus a Twilio-style .messages.create in a file without anthropic
    assert not [s for s in scan.sites if s.file == "shopdesk/notify/sms.py"]


def test_templates_are_str_format_compatible(scan):
    for s in scan.sites:
        if s.template:
            values = {v: f"<{v}>" for v in s.template.variables}
            s.template.user.format(**values)
            (s.template.system or "").format(**values)
    extract = next(s for s in scan.sites if s.id == "cs_10")
    assert (
        extract.template.user.format(email="hi")
        == 'Email:\nhi\n\nReturn JSON only, e.g. {"order_id": "A-1"}.'
    )


CALL = "client.chat.completions.create(model='m', messages=[{'role': 'user', 'content': 'hi'}])\n"
OPENAI_FILE = "from openai import OpenAI\nclient = OpenAI()\n" + CALL


def test_walk_skips_tests_hidden_venv_and_unparseable(tmp_path):
    files = {
        "app/main.py": OPENAI_FILE,
        "app/broken.py": "def (:\n",
        "tests/test_main.py": OPENAI_FILE,
        "app/tests/helper.py": OPENAI_FILE,
        ".hidden/x.py": OPENAI_FILE,
        "venv/lib/x.py": OPENAI_FILE,
        "pyenv312/lib/x.py": OPENAI_FILE,
        "pyenv312/pyvenv.cfg": "home = /usr/bin\n",
    }
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    scan = scan_repo(tmp_path)
    assert [s.file for s in scan.sites] == ["app/main.py"]
    assert scan.warnings == ["skipped app/broken.py: not parseable as Python 3.12 (SyntaxError)"]


def test_client_tracking_rejects_wrong_sdk_and_accepts_inline_constructor(tmp_path):
    (tmp_path / "a.py").write_text(
        "import anthropic\nfrom openai import OpenAI\n"
        "claude = anthropic.Anthropic()\n"
        "claude.chat.completions.create(model='m', messages=[])\n"  # anthropic client: not an openai call
        "OpenAI().chat.completions.create(model='m', messages=[{'role': 'user', 'content': 'hi'}])\n"
    )
    [site] = scan_repo(tmp_path).sites
    assert (site.lines, site.call, site.prompt_source) == (
        [5, 5],
        "OpenAI().chat.completions.create",
        "literal",
    )
    assert site.template == tpl("hi")


def _scan_module(tmp_path, source: str) -> list[CallSite]:
    (tmp_path / "m.py").write_text(source)
    return scan_repo(tmp_path).sites


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "from openai import OpenAI as OA\nc = OA()\nc.chat.completions.create(model='m', messages=[])\n",
            [("c.chat.completions.create", "openai")],
        ),
        (
            "import openai as oai\nc = oai.OpenAI()\n\n\ndef f():\n"
            "    c.chat.completions.create(model='m', messages=[])\n"
            "    oai.chat.completions.create(model='m', messages=[])\n",
            [("c.chat.completions.create", "openai"), ("oai.chat.completions.create", "openai")],
        ),
        (
            "from openai import AsyncOpenAI\nac = AsyncOpenAI()\n\n\nasync def f():\n"
            "    return await ac.chat.completions.create(model='m', messages=[])\n",
            [("ac.chat.completions.create", "openai")],
        ),
        (  # a parameter shadowing the module name is an injected client
            "import openai\n\n\ndef f(openai):\n"
            "    return openai.chat.completions.create(model='m', messages=[])\n",
            [("openai.chat.completions.create", "openai")],
        ),
        (
            "from openai import OpenAI\nwith OpenAI() as c:\n"
            "    c.chat.completions.create(model='m', messages=[])\n",
            [("c.chat.completions.create", "openai")],
        ),
        (  # an untyped parameter shadows a module-level client of the other SDK
            "import openai\nfrom anthropic import Anthropic\nclient = Anthropic()\n\n\ndef f(client):\n"
            "    return client.chat.completions.create(model='m', messages=[])\n",
            [("client.chat.completions.create", "openai")],
        ),
    ],
    ids=[
        "from-import-alias",
        "module-alias",
        "async-await",
        "shadowed-module-name",
        "with-as",
        "param-shadow",
    ],
)
def test_import_and_binding_shapes(tmp_path, source, expected):
    assert [(s.call, s.sdk) for s in _scan_module(tmp_path, source)] == expected


FSTRING_MODULE = """\
from openai import OpenAI

client = OpenAI()
{prelude}


def f(i, s, price, req, req_id, x, width):
    return client.chat.completions.create(model="m", messages=[{{"role": "user", "content": {body}}}])
"""


@pytest.mark.parametrize(
    ("prelude", "body", "user", "variables", "specs"),
    [
        ("", 'f"Item {i + 1}: {s}"', "Item {i_1}: {s}", ["i_1", "s"], {}),
        ("", 'f"Total {1 + 1}: {s}"', "Total 2: {s}", ["s"], {}),
        ("N = 3", 'f"Top {N * 2} of {s}"', "Top 6 of {s}", ["s"], {}),
        ("CAP = 5", 'f"{CAP:03d} {s}"', "005 {s}", ["s"], {}),
        ('QUEUES = ["a", "b"]', 'f"Pick one of {QUEUES}: {s}"', "Pick one of ['a', 'b']: {s}", ["s"], {}),
        (
            'LIMIT = 10000\nRULE = f"At most {LIMIT:,} chars"',
            'f"{RULE}. {s}"',
            "At most 10,000 chars. {s}",
            ["s"],
            {},
        ),
        ("", 'f"{price:.2f} {s!r}"', "{price} {s}", ["price", "s"], {"price": ":.2f", "s": "!r"}),
        ("", 'f"{s!r:>10}"', "{s}", ["s"], {"s": "!r:>10"}),
        (
            "",
            '"{a:>8} {b!s}".format(a=price, b=s)',
            "{price} {s}",
            ["price", "s"],
            {"price": ":>8", "s": "!s"},
        ),
        ("WIDTH = 4", '"{a:0{w}d}".format(a=7, w=WIDTH)', None, None, None),  # nested spec: dynamic
        ("", 'f"{req.id} {req_id}"', None, None, None),  # two expressions, one placeholder name
        ("", 'f"{x} {x:.2f}"', None, None, None),  # one placeholder, two formats
        ("", 'f"{x:{width}}"', None, None, None),  # runtime format spec
    ],
)
def test_fstring_fields(tmp_path, prelude, body, user, variables, specs):
    [site] = _scan_module(tmp_path, FSTRING_MODULE.format(prelude=prelude, body=body))
    if user is None:
        assert (site.prompt_source, site.template, site.flags) == ("dynamic", None, ["NEEDS_RUNTIME_CAPTURE"])
        return
    assert (site.template.user, site.template.variables) == (user, variables)
    assert site.params.get("template_format_specs", {}) == specs


MUTATION_MODULE = """\
from openai import OpenAI

client = OpenAI()
SYSTEM = "base"
TOOLS = [{{"name": "a"}}]
RF = {{"type": "json_object"}}
{mutation}


def f(q):
    return client.chat.completions.create(
        model="m",
        messages=[{{"role": "system", "content": SYSTEM}}, {{"role": "user", "content": q}}],
        tools=TOOLS,
        response_format=RF,
    )
"""
X = "<expr>"


@pytest.mark.parametrize(
    ("mutation", "tools", "rf", "system"),
    [
        ("", [{"name": "a"}], {"type": "json_object"}, "base"),
        ('TOOLS.append({"name": "b"})', X, {"type": "json_object"}, "base"),
        ("\n\ndef g():\n    TOOLS.extend([])", X, {"type": "json_object"}, "base"),
        ("\n\ndef g():\n    return TOOLS.pop()", X, {"type": "json_object"}, "base"),
        ('TOOLS[0]["name"] = "c"', X, {"type": "json_object"}, "base"),
        ('RF["type"] = "json_schema"', [{"name": "a"}], X, "base"),
        ('del RF["type"]', [{"name": "a"}], X, "base"),
        (
            '\n\ndef g():\n    global SYSTEM\n    SYSTEM = "override"',
            [{"name": "a"}],
            {"type": "json_object"},
            None,
        ),
    ],
    ids=["none", "append", "extend-in-fn", "pop-in-fn", "nested-item", "item", "del-item", "global-rebind"],
)
def test_mutated_module_constants_stay_unresolved(tmp_path, mutation, tools, rf, system):
    [site] = _scan_module(tmp_path, MUTATION_MODULE.format(mutation=mutation))
    assert (site.params["tools"], site.params["response_format"]) == (tools, rf)
    assert (site.template.system if site.template else None) == system


def test_bom_and_coding_cookie_files_parse(tmp_path):
    (tmp_path / "bom.py").write_bytes(b"\xef\xbb\xbf" + OPENAI_FILE.encode())
    (tmp_path / "latin1.py").write_bytes(b"# -*- coding: latin-1 -*-\n# caf\xe9\n" + OPENAI_FILE.encode())
    assert [s.file for s in scan_repo(tmp_path).sites] == ["bom.py", "latin1.py"]


def test_recursion_error_skips_only_that_file(tmp_path, caplog):
    (tmp_path / "deep.py").write_text("x = " + " + ".join(["1"] * 3000) + "\n")
    (tmp_path / "ok.py").write_text(OPENAI_FILE)
    with caplog.at_level("WARNING", logger="offramp.agents.scanner"):
        scan = scan_repo(tmp_path)
    assert [s.file for s in scan.sites] == ["ok.py"]
    assert scan.warnings == ["skipped deep.py: nested too deeply to analyse (RecursionError)"]
    assert "deep.py" in caplog.text
