"""Target-repo code that Offramp imports (Pydantic schemas, the refund oracle) runs without Offramp's
secrets: no provider key, token, fallback key or LangSmith setting reaches its environment."""

import json

from offramp.agents import case_gen
from offramp.schemas import CallSite, PromptTemplate

PROBE = """import json, os, pathlib

from pydantic import BaseModel

pathlib.Path(__file__).with_name("seen.json").write_text(json.dumps(sorted(os.environ)))


class Order(BaseModel):
    order_id: str
"""
SECRETS = {
    "NEBIUS_API_KEY": "nb-secret",
    "NEBIUS_API_KEY_FALLBACK": "nb-fallback",
    "TAVILY_API_KEY": "tvly-secret",
    "LANGSMITH_API_KEY": "ls-secret",
    "LANGSMITH_TRACING": "true",
    "TENDEM_API_KEY_FALLBACK": "td-fallback",
    "GITHUB_TOKEN": "ghp-secret",
    "SOME_SERVICE_TOKEN": "x-secret",
}


def test_a_target_module_imported_for_its_schema_sees_no_secret(tmp_path, monkeypatch):
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("OFFRAMP_DATA_DIR", str(tmp_path / "data"))
    repo = tmp_path / "target"
    (repo / "shop").mkdir(parents=True)
    (repo / "shop" / "__init__.py").write_text("")
    (repo / "shop" / "orders.py").write_text(PROBE)
    site = CallSite(
        id="cs_01",
        file="shop/orders.py",
        lines=[1, 2],
        sdk="openai",
        call="client.chat.completions.create",
        prompt_source="literal",
        template=PromptTemplate(user="Extract the order as <pydantic:Order>"),
    )

    schemas = case_gen.resolved_schemas(site, repo)

    assert schemas["Order"]["properties"] == {"order_id": {"title": "Order Id", "type": "string"}}
    seen = set(json.loads((repo / "shop" / "seen.json").read_text()))
    assert not seen & set(SECRETS)
    assert not [name for name in seen if any(s in name for s in ("KEY", "TOKEN", "LANGSMITH", "FALLBACK"))]
    assert {"PATH", "HOME", "PYTHONPATH"} <= seen
