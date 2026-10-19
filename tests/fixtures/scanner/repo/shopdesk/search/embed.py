"""Embed KB articles for search (fixture: embeddings call, never migrated)."""

from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
client = OpenAI()


def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]
