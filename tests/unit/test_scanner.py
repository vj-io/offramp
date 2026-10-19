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
    CallSite(
        id="cs_12",
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
        id="cs_13",
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
        id="cs_14",
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
    assert scan.repo == str(REPO)
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
    assert [s.file for s in scan_repo(tmp_path).sites] == ["app/main.py"]


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
