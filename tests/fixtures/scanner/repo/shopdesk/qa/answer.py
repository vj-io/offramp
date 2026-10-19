"""Answer from knowledge-base snippets (fixture: injected client without annotation, prompt in a local f-string)."""

import openai

QA_SYSTEM = "Answer using only the numbered knowledge-base snippets and cite them like [2]."


def answer(question: str, snippets: list[str], client) -> str:
    context = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(snippets, 1))
    user_msg = f"Snippets:\n{context}\n\nQuestion: {question}"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": QA_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
    except openai.APIError:
        return "Sorry, something went wrong."
    return resp.choices[0].message.content
