"""RAG failure triage.

Diagnose *why* a query is failing before tuning the embedding model, chunk
size, or prompt. Four checks, in the order you should run them:

1. irrelevant_documents  -- is the answer even in the corpus / retrieved set?
2. answer_position       -- right docs, wrong answer: is it lost in the middle?
3. phrasing_sensitivity  -- right sometimes, wrong sometimes: query formulation?
4. latency_breakdown     -- everything is slow: which stage is the bottleneck?
"""
import time
from dataclasses import dataclass, field

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

from .opensearch_client import bm25_search, hybrid_search, knn_search
from .reranker import rerank


# ── 1. Irrelevant documents ──────────────────────────────────────────────────


@dataclass
class IrrelevantDocsReport:
    query: str
    vector_hits: list[dict]
    bm25_hits: list[dict]
    vector_found_answer: bool | None
    bm25_found_answer: bool | None
    diagnosis: str


def diagnose_irrelevant_documents(
    client: OpenSearch,
    embed_model: SentenceTransformer,
    query: str,
    answer_substring: str | None = None,
    k: int = 10,
) -> IrrelevantDocsReport:
    """Pull top-k for both retrieval paths and check whether the answer surfaces.

    Pass `answer_substring` (a snippet you know should be in the corpus) to get
    an automatic verdict; otherwise inspect `.vector_hits` / `.bm25_hits` yourself.
    """
    query_vec = embed_model.encode(query, normalize_embeddings=True).tolist()
    vector_hits = knn_search(client, query_vec, k=k)
    bm25_hits = bm25_search(client, query, k=k)

    vector_found = bm25_found = None
    if answer_substring:
        needle = answer_substring.lower()
        vector_found = any(needle in h["chunk_text"].lower() for h in vector_hits)
        bm25_found = any(needle in h["chunk_text"].lower() for h in bm25_hits)

    if answer_substring is None:
        diagnosis = "No answer_substring given — inspect vector_hits/bm25_hits manually."
    elif vector_found:
        diagnosis = "Vector search finds it. Retrieval is not the problem."
    elif bm25_found and not vector_found:
        diagnosis = "BM25 finds it but vector search doesn't -> investigate embeddings (model choice, chunking, normalization)."
    else:
        diagnosis = "Neither BM25 nor vector search finds it -> stop tuning retrieval, the problem is ingestion (the chunk isn't indexed, or it's split across chunks)."

    return IrrelevantDocsReport(
        query=query,
        vector_hits=vector_hits,
        bm25_hits=bm25_hits,
        vector_found_answer=vector_found,
        bm25_found_answer=bm25_found,
        diagnosis=diagnosis,
    )


# ── 2. Right documents, wrong answer ─────────────────────────────────────────


@dataclass
class AnswerPositionReport:
    query: str
    hits: list[dict]
    answer_rank: int | None  # 1-indexed position among hits, None if not found
    reranked_hits: list[dict]
    reranked_answer_rank: int | None
    diagnosis: str


def diagnose_answer_position(
    client: OpenSearch,
    embed_model: SentenceTransformer,
    query: str,
    answer_substring: str,
    k: int = 10,
) -> AnswerPositionReport:
    """Locate the answer within the retrieved context and see if reranking helps.

    Buried-in-the-middle results are a classic "lost in the middle" failure:
    the LLM has the right context but ignores/misses content in the middle
    of a long prompt.
    """
    query_vec = embed_model.encode(query, normalize_embeddings=True).tolist()
    hits = hybrid_search(client, query, query_vec, k=k)
    needle = answer_substring.lower()

    answer_rank = next(
        (i for i, h in enumerate(hits, 1) if needle in h["chunk_text"].lower()), None
    )

    reranked = rerank(query, list(hits), top_n=k)
    reranked_rank = next(
        (i for i, h in enumerate(reranked, 1) if needle in h["chunk_text"].lower()), None
    )

    if answer_rank is None:
        diagnosis = "Answer not present in the retrieved set at all -> this is an irrelevant_documents problem, not a position problem."
    elif answer_rank <= 2:
        diagnosis = "Answer is already near the top. Position is not the issue -- look at the prompt/generation step."
    elif reranked_rank is not None and reranked_rank < answer_rank:
        diagnosis = f"Answer was buried at rank {answer_rank}; reranking moved it to {reranked_rank}. Add a reranking stage."
    else:
        diagnosis = f"Answer buried at rank {answer_rank} and reranking didn't help much -> consider fewer/shorter chunks per prompt so nothing gets lost in the middle."

    return AnswerPositionReport(
        query=query,
        hits=hits,
        answer_rank=answer_rank,
        reranked_hits=reranked,
        reranked_answer_rank=reranked_rank,
        diagnosis=diagnosis,
    )


# ── 3. Right sometimes, wrong sometimes ──────────────────────────────────────


@dataclass
class PhrasingSensitivityReport:
    base_query: str
    variants: list[str]
    hit_sets: list[list[dict]]
    overlap_ratio: float  # fraction of top-k chunk ids shared across all variants
    diagnosis: str


def diagnose_phrasing_sensitivity(
    client: OpenSearch,
    embed_model: SentenceTransformer,
    variants: list[str],
    k: int = 5,
) -> PhrasingSensitivityReport:
    """Run near-duplicate phrasings of the same question and compare retrieved sets.

    Low overlap across phrasings that mean the same thing points at query
    formulation (short/ambiguous queries embed poorly), not a documents problem.
    """
    if len(variants) < 2:
        raise ValueError("Need at least two phrasings to compare")

    hit_sets = []
    for q in variants:
        vec = embed_model.encode(q, normalize_embeddings=True).tolist()
        hit_sets.append(hybrid_search(client, q, vec, k=k))

    id_sets = [
        {(h["episode_title"], h["chunk_index"]) for h in hits} for hits in hit_sets
    ]
    common = set.intersection(*id_sets)
    union = set.union(*id_sets)
    overlap_ratio = len(common) / len(union) if union else 0.0

    if overlap_ratio >= 0.7:
        diagnosis = "Retrieved sets are stable across phrasings. Query formulation is not the issue."
    elif overlap_ratio >= 0.3:
        diagnosis = "Moderate drift across phrasings -> consider query expansion/rewriting before retrieval."
    else:
        diagnosis = "Retrieved sets barely overlap across equivalent phrasings -> query formulation is likely the root cause, not the documents."

    return PhrasingSensitivityReport(
        base_query=variants[0],
        variants=variants,
        hit_sets=hit_sets,
        overlap_ratio=overlap_ratio,
        diagnosis=diagnosis,
    )


# ── 4. Everything is too slow ────────────────────────────────────────────────


@dataclass
class LatencyBreakdown:
    query: str
    stage_ms: dict[str, float] = field(default_factory=dict)
    bottleneck: str = ""
    diagnosis: str = ""


def diagnose_latency(
    client: OpenSearch,
    embed_model: SentenceTransformer,
    query: str,
    ask_fn,
    k: int = 5,
    use_reranking: bool = True,
) -> LatencyBreakdown:
    """Time each pipeline stage separately instead of guessing where time goes.

    `ask_fn` is a callable `(question, hits) -> str`, e.g. `src.rag.ask`, so
    generation latency (including the network hop to the LLM) is included.
    """
    stages: dict[str, float] = {}

    t0 = time.perf_counter()
    query_vec = embed_model.encode(query, normalize_embeddings=True).tolist()
    stages["embedding"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    hits = hybrid_search(client, query, query_vec, k=k)
    stages["retrieval"] = (time.perf_counter() - t0) * 1000

    if use_reranking:
        t0 = time.perf_counter()
        hits = rerank(query, hits, top_n=k)
        stages["reranking"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    ask_fn(query, hits)
    stages["generation"] = (time.perf_counter() - t0) * 1000

    bottleneck = max(stages, key=stages.get)
    total = sum(stages.values())
    diagnosis = (
        f"'{bottleneck}' is the largest slice ({stages[bottleneck]:.0f}ms of "
        f"{total:.0f}ms total) -> optimize that stage first, not the vector database."
    )

    return LatencyBreakdown(query=query, stage_ms=stages, bottleneck=bottleneck, diagnosis=diagnosis)
