"""Answer a customer question from retrieved knowledge-base articles."""

from openai import OpenAI
from pydantic import BaseModel

from helphub.kb import Article

RAG_SYSTEM_PROMPT = """You are HelpHub's support assistant. Answer the customer's question \
using only the knowledge-base excerpts provided in the user message.

- Cite every fact with the excerpt id in square brackets, for example [refund-policy].
- If the excerpts do not answer the question, say you do not have that information and offer to \
connect the customer with a human agent. Do not guess.
- Reply in the language the customer wrote in, in at most 120 words."""


class Answer(BaseModel):
    text: str
    sources: list[str]  # ids of the excerpts the answer cites


def answer_question(client: OpenAI, question: str, articles: list[Article]) -> Answer:
    excerpts = "\n\n".join(f"[{a.id}] {a.title}\n{a.text}" for a in articles) or "(none found)"
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        max_tokens=400,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Knowledge-base excerpts:\n{excerpts}\n\nCustomer question: {question}",
            },
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    return Answer(text=text, sources=[a.id for a in articles if f"[{a.id}]" in text])
