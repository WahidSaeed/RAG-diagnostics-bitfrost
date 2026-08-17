"""RAG pipeline: retrieve relevant chunks then generate an answer via Bifrost."""
import os
from openai import OpenAI


DEFAULT_MODEL = os.getenv("GROQ_MODEL", "groq/openai/gpt-oss-120b")
FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "groq/openai/gpt-oss-20b")
BIFROST_BASE_URL = os.getenv("BIFROST_BASE_URL", "http://localhost:8080/v1")
BIFROST_CACHE_KEY = "vector-podcast-rag"


def get_bifrost_client() -> OpenAI:
    return OpenAI(base_url=BIFROST_BASE_URL, api_key="bifrost")


def build_context(hits: list[dict], max_chars: int = 6000) -> str:
    """Format retrieved chunks into a readable context block."""
    parts = []
    total = 0
    for i, hit in enumerate(hits, 1):
        snippet = (
            f"[{i}] Episode: {hit['episode_title']}\n"
            f"    Score: {hit['score']:.4f}\n"
            f"    Text: {hit['chunk_text']}\n"
        )
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n".join(parts)


SYSTEM_PROMPT = """You are a helpful assistant with deep expertise in vector search, \
embeddings, and AI-powered search. You answer questions using the podcast transcript \
excerpts provided as context.

Rules:
- Answer only from the provided context. If the context doesn't contain enough \
information, say so clearly.
- Cite the episode title when you reference specific content.
- Be concise but complete. Use bullet points for lists.
- If asked for opinions, frame them as "according to the podcast guests"."""


def ask(
    question: str,
    hits: list[dict],
    model: str | None = None,
) -> str:
    """Generate an answer grounded in the retrieved chunks."""
    client = get_bifrost_client()
    context = build_context(hits)
    model = model or DEFAULT_MODEL

    user_message = f"""Context from Vector Podcast transcripts:

{context}

---
Question: {question}"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        extra_body={"fallbacks": [FALLBACK_MODEL]},
        extra_headers={"x-bf-cache-key": BIFROST_CACHE_KEY},
    )
    return response.choices[0].message.content


def ask_streaming(
    question: str,
    hits: list[dict],
    model: str | None = None,
) -> None:
    """Stream the answer to stdout."""
    client = get_bifrost_client()
    context = build_context(hits)
    model = model or DEFAULT_MODEL

    user_message = f"""Context from Vector Podcast transcripts:

{context}

---
Question: {question}"""

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        stream=True,
        extra_body={"fallbacks": [FALLBACK_MODEL]},
        extra_headers={"x-bf-cache-key": BIFROST_CACHE_KEY},
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()
