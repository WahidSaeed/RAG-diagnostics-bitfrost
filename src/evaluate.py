"""RAGAS evaluation of the RAG pipeline against a labeled question set.

Unlike diagnostics.py (which triages a single failing query), this scores the
whole pipeline so you can regression-test changes to chunk size, embedding
model, or reranking against a fixed baseline.

Every run emits structured telemetry (one JSON record per event: retrieval,
generation, and each individual RAGAS metric judgment — start, latency,
result/error) to logs/evaluate-<run_id>.jsonl, and optionally forwards each
record live via an `on_event` callback (used by the API to stream progress
and logs to the frontend).

Usage:
    uv run python -m src.evaluate
"""
import functools
import json
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

EventCallback = Callable[[dict], None]

# ragas unconditionally calls nest_asyncio.apply() at import time (for Jupyter
# compatibility). On Python 3.14 that patch breaks asyncio.timeout()'s task
# tracking ("Timeout should be used inside a task"), which surfaces as every
# ragas metric silently failing. We never call evaluate() from inside an
# already-running event loop, so the patch isn't needed here — stub it out
# before ragas imports it.
_nest_asyncio_stub = types.ModuleType("nest_asyncio")
_nest_asyncio_stub.apply = lambda *_args, **_kwargs: None
sys.modules.setdefault("nest_asyncio", _nest_asyncio_stub)

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from opensearchpy import OpenSearch
from ragas import EvaluationDataset, evaluate
from ragas.evaluation import EvaluationResult
from ragas.run_config import RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from sentence_transformers import SentenceTransformer

from .opensearch_client import get_client, hybrid_search
from .rag import BIFROST_BASE_URL, BIFROST_CACHE_KEY, DEFAULT_MODEL, FALLBACK_MODELS, ask

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


class Telemetry:
    """Records structured evaluation events to a JSONL trace file and forwards
    each one to an optional live callback (the API streams these to the UI)."""

    def __init__(self, on_event: EventCallback | None = None):
        LOGS_DIR.mkdir(exist_ok=True)
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        self.path = LOGS_DIR / f"evaluate-{self.run_id}.jsonl"
        self._file = self.path.open("a")
        self._t0 = time.perf_counter()
        self._on_event = on_event

    def log(self, level: str, event: str, **fields: Any) -> None:
        record = {
            "ts": round(time.perf_counter() - self._t0, 3),
            "level": level,
            "event": event,
            **fields,
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        if self._on_event:
            self._on_event(record)

    def progress(self, phase: str, current: int, total: int) -> None:
        self.log("info", "progress", phase=phase, current=current, total=total)

    def close(self) -> None:
        self._file.close()


class SentenceTransformerEmbeddings(Embeddings):
    """Adapts an already-loaded SentenceTransformer to RAGAS's embeddings interface,
    so callers that already hold a loaded model (e.g. the FastAPI backend) don't
    have to load a second copy just for judging."""

    def __init__(self, model: SentenceTransformer):
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()


class _RagasProgressBar:
    """Duck-types tqdm's `.update(n)` — RAGAS's Executor calls only that method
    when a custom `_pbar` is supplied to evaluate(), so this lets us report
    per-judgment progress through a plain callback instead of a real tqdm bar."""

    def __init__(self, total: int, on_update: Callable[[int, int], None]):
        self.total = total
        self._count = 0
        self._on_update = on_update

    def update(self, n: int = 1) -> None:
        self._count += n
        self._on_update(self._count, self.total)


def _instrument_metric(metric, telemetry: Telemetry):
    """Wrap a metric's single_turn_ascore so every individual judgment (one
    question x one metric) logs its own start/latency/score/error — the level
    of detail RAGAS's own progress bar doesn't expose."""
    original = metric.single_turn_ascore

    @functools.wraps(original)
    async def instrumented(sample, callbacks=None, timeout=None):
        question = getattr(sample, "user_input", "?")
        t0 = time.perf_counter()
        telemetry.log("debug", "metric_start", metric=metric.name, question=question)
        try:
            score = await original(sample, callbacks, timeout=timeout)
            telemetry.log(
                "info",
                "metric_done",
                metric=metric.name,
                question=question,
                score=score,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            )
            return score
        except Exception as exc:
            telemetry.log(
                "error",
                "metric_error",
                metric=metric.name,
                question=question,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            )
            raise

    metric.single_turn_ascore = instrumented
    return metric


@dataclass
class EvalExample:
    question: str
    ground_truth: str


# A small, hand-labeled question set grounded in specific Podcast Transcripts
# episodes, so `reference` is checkable against them.
EVAL_SET: list[EvalExample] = [
    EvalExample(
        question="What is HNSW and who is credited with inventing it?",
        ground_truth=(
            "HNSW (Hierarchical Navigable Small World) is a graph-based "
            "approximate nearest-neighbour search algorithm. It was invented "
            "by Yury Malkov, who has worked as a staff engineer at Twitter "
            "and is described as the author of the most widely adopted ANN "
            "algorithm."
        ),
    ),
    EvalExample(
        question="What are wormhole vectors and who introduced the idea?",
        ground_truth=(
            "Wormhole vectors are a concept introduced by Trey Grainger for "
            "traversing between disparate vector spaces (e.g. sparse, dense, "
            "and behavioral) to improve hybrid search, drawing an analogy "
            "from physics."
        ),
    ),
    EvalExample(
        question="What ANN libraries or algorithms are commonly compared to HNSW?",
        ground_truth=(
            "Common comparisons include ANNOY (Spotify's approximate nearest "
            "neighbour library), FAISS (Facebook AI Similarity Search), and "
            "HNSW/hnswlib."
        ),
    ),
    EvalExample(
        question="What is the role of sparse vectors in hybrid search?",
        ground_truth=(
            "Sparse vectors capture exact lexical/term-based matching "
            "(similar to BM25), and are combined with dense vectors in "
            "hybrid search to cover both keyword precision and semantic "
            "similarity."
        ),
    ),
]


def build_dataset(
    top_k: int = 5,
    client: OpenSearch | None = None,
    embed_model: SentenceTransformer | None = None,
    telemetry: Telemetry | None = None,
    examples: list[EvalExample] | None = None,
) -> EvaluationDataset:
    """Run each eval question through the live retrieval + generation pipeline."""
    client = client or get_client()
    embed_model = embed_model or SentenceTransformer("all-MiniLM-L6-v2")
    examples = examples or EVAL_SET

    rows = []
    for i, example in enumerate(examples):
        t0 = time.perf_counter()
        query_vec = embed_model.encode(example.question, normalize_embeddings=True).tolist()
        hits = hybrid_search(client, example.question, query_vec, k=top_k)
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)
        if telemetry:
            telemetry.log(
                "info",
                "retrieval_done",
                question=example.question,
                num_hits=len(hits),
                latency_ms=retrieval_ms,
            )

        t0 = time.perf_counter()
        contexts = [h["chunk_text"] for h in hits]
        answer = ask(example.question, hits)
        generation_ms = round((time.perf_counter() - t0) * 1000, 1)
        if telemetry:
            telemetry.log(
                "info",
                "generation_done",
                question=example.question,
                answer_chars=len(answer),
                latency_ms=generation_ms,
            )

        rows.append(
            {
                "user_input": example.question,
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": example.ground_truth,
            }
        )
        if telemetry:
            telemetry.progress("retrieving", i + 1, len(examples))

    return EvaluationDataset.from_list(rows)


def run_evaluation(
    top_k: int = 5,
    client: OpenSearch | None = None,
    embed_model: SentenceTransformer | None = None,
    telemetry: Telemetry | None = None,
    examples: list[EvalExample] | None = None,
) -> EvaluationResult:
    """Score the pipeline with RAGAS: faithfulness, relevancy, context precision/recall."""
    embed_model = embed_model or SentenceTransformer("all-MiniLM-L6-v2")
    dataset = build_dataset(
        top_k=top_k, client=client, embed_model=embed_model, telemetry=telemetry, examples=examples
    )

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            base_url=BIFROST_BASE_URL,
            api_key="bifrost",
            model=DEFAULT_MODEL,
            # Same Bifrost fallback used for chat generation (src/rag.py): if the
            # primary model is rate-limited, retry against the fallback instead
            # of failing every judgment outright.
            extra_body={"fallbacks": FALLBACK_MODELS},
            default_headers={"x-bf-cache-key": BIFROST_CACHE_KEY},
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(SentenceTransformerEmbeddings(embed_model))
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]
    if telemetry:
        metrics = [_instrument_metric(m, telemetry) for m in metrics]

    pbar = (
        _RagasProgressBar(
            total=len(dataset) * len(metrics),
            on_update=lambda current, total: telemetry.progress("judging", current, total),
        )
        if telemetry
        else None
    )

    return evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        # Faithfulness/context-precision issue several sequential sub-calls per
        # sample; running many samples concurrently through a single Groq key
        # queues past the default timeout, so keep concurrency low and the
        # per-call budget generous.
        run_config=RunConfig(max_workers=2, timeout=300),
        _pbar=pbar,
    )


def run_evaluation_json(
    top_k: int = 5,
    client: OpenSearch | None = None,
    embed_model: SentenceTransformer | None = None,
    on_event: EventCallback | None = None,
    examples: list[EvalExample] | None = None,
) -> dict:
    """Same as run_evaluation, flattened to JSON-safe types for the API layer.

    Every retrieval/generation/judgment event is logged to a JSONL trace file
    under logs/; on_event (if given) also receives each record live, and the
    trace file path is included in the returned dict. Pass `examples` to
    evaluate a custom question set instead of the default EVAL_SET.
    """
    examples = examples or EVAL_SET
    telemetry = Telemetry(on_event=on_event)
    telemetry.log("info", "run_start", top_k=top_k, questions=len(examples))
    try:
        df = run_evaluation(
            top_k=top_k, client=client, embed_model=embed_model, telemetry=telemetry, examples=examples
        ).to_pandas()
    finally:
        telemetry.log("info", "run_done")
        telemetry.close()

    metric_cols = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]
    mean_scores = {col: (None if df[col].isna().all() else float(df[col].mean())) for col in metric_cols}

    df = df.where(df.notna(), None)  # NaN isn't valid JSON

    rows = [
        {
            "question": row["user_input"],
            "response": row["response"],
            "reference": row["reference"],
            "faithfulness": row["faithfulness"],
            "answer_relevancy": row["answer_relevancy"],
            "context_precision": row["llm_context_precision_with_reference"],
            "context_recall": row["context_recall"],
        }
        for _, row in df.iterrows()
    ]

    return {"rows": rows, "mean_scores": mean_scores, "log_file": str(telemetry.path)}


if __name__ == "__main__":
    telemetry = Telemetry(on_event=lambda r: print(json.dumps(r)))
    try:
        result = run_evaluation(telemetry=telemetry)
    finally:
        telemetry.close()
    df = result.to_pandas()
    print(df.to_string())
    print("\nMean scores:")
    print(df.select_dtypes("number").mean())
    print(f"\nTrace: {telemetry.path}")
