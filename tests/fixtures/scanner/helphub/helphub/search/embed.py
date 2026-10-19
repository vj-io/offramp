"""Semantic search over the knowledge base using OpenAI embeddings."""

import math

from openai import OpenAI

from helphub.kb import Article


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def cosine(a: list[float], b: list[float]) -> float:
    norm = math.hypot(*a) * math.hypot(*b)
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


def search_kb(client: OpenAI, query: str, articles: list[Article], k: int = 3) -> list[Article]:
    if not query.strip() or not articles:
        return []
    query_vec, *article_vecs = embed_texts(client, [query] + [a.text for a in articles])
    ranked = sorted(
        zip(articles, article_vecs, strict=True), key=lambda pair: -cosine(query_vec, pair[1])
    )
    return [article for article, _ in ranked[:k]]
