"""Cross-encoder re-ranking for improving retrieval precision."""
from sentence_transformers import CrossEncoder


_model_cache: dict[str, CrossEncoder] = {}


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    if model_name not in _model_cache:
        _model_cache[model_name] = CrossEncoder(model_name)
    return _model_cache[model_name]


def rerank(
    query: str,
    hits: list[dict],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int | None = None,
) -> list[dict]:
    """Re-score hits with a cross-encoder and return them sorted by relevance.

    Cross-encoders process (query, passage) pairs jointly through a transformer,
    producing much more accurate relevance scores than cosine similarity of
    independently-encoded embeddings, at the cost of being too slow for
    first-stage retrieval over the full index.
    """
    if not hits:
        return hits

    reranker = get_reranker(model_name)

    pairs = [(query, hit["chunk_text"]) for hit in hits]
    scores = reranker.predict(pairs)

    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)
        hit["original_score"] = hit["score"]

    reranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)

    if top_n is not None:
        reranked = reranked[:top_n]

    return reranked
