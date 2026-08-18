"""RAG pipeline: retrieve relevant chunks then generate an answer via Bifrost."""
import os
from openai import OpenAI


# Bifrost routes "<provider>/<model>" and strips the leading "groq/" as the
# provider name — but Groq's actual model id for these two IS "groq/compound"
# / "groq/compound-mini" (the prefix is baked into the id itself, confirmed
# directly against api.groq.com). So through Bifrost they need the doubled
# "groq/groq/compound" form, or Bifrost forwards a bare "compound" that Groq
# 404s on.
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "groq/groq/compound")
FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "groq/groq/compound-mini")
# Tried in order after DEFAULT_MODEL. compound/compound-mini are kept as the
# primary/first-fallback pair per priority; the gpt-oss models are further
# fallbacks. Note: compound is Groq's agentic model and internally routes
# part of its pipeline through openai/gpt-oss-120b, so it can still fail if
# that specific model is rate-limited org-wide, even though compound itself
# has no separate daily token cap.
FALLBACK_MODELS = [FALLBACK_MODEL, "groq/openai/gpt-oss-120b", "groq/openai/gpt-oss-20b"]
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
        extra_body={"fallbacks": FALLBACK_MODELS},
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
        extra_body={"fallbacks": FALLBACK_MODELS},
        extra_headers={"x-bf-cache-key": BIFROST_CACHE_KEY},
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()
