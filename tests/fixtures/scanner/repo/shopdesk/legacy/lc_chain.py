"""LangChain chat models (fixture: ChatOpenAI / ChatAnthropic constructors; prompts arrive later via invoke)."""

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)


def make_fast_llm(max_tokens: int = 256) -> ChatAnthropic:
    return ChatAnthropic(model_name="claude-3-5-haiku-20241022", max_tokens=max_tokens)
