"""FastAPI backend for the Vector Podcast RAG app, consumed by the Next.js frontend."""
import json
import os
import queue
import threading

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src.opensearch_client import get_client, knn_search, hybrid_search, INDEX_NAME
from src.rag import (
    SYSTEM_PROMPT,
    build_context,
    DEFAULT_MODEL,
    FALLBACK_MODELS,
    BIFROST_CACHE_KEY,
    get_bifrost_client,
    ask,
)
from src.diagnostics import (
    diagnose_irrelevant_documents,
    diagnose_answer_position,
    diagnose_phrasing_sensitivity,
    diagnose_latency,
)
from src.evaluate import run_evaluation_json, EvalExample, EVAL_SET

app = FastAPI(title="Vector Podcast RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")],
    allow_methods=["*"],
    allow_headers=["*"],
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
os_client = get_client()
bifrost_client = get_bifrost_client()


class ChatRequest(BaseModel):
    question: str
    search_mode: str = "hybrid"  # "hybrid" | "semantic"
    top_k: int = 5
    model: str | None = None


class DiagnoseIrrelevantRequest(BaseModel):
    query: str
    answer_substring: str | None = None
    top_k: int = 10


class DiagnoseAnswerPositionRequest(BaseModel):
    query: str
    answer_substring: str
    top_k: int = 10


class DiagnosePhrasingRequest(BaseModel):
    variants: list[str]
    top_k: int = 5


class DiagnoseLatencyRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranking: bool = True


class EvalExampleRequest(BaseModel):
    question: str
    ground_truth: str


class EvaluateRequest(BaseModel):
    top_k: int = 5
    examples: list[EvalExampleRequest] | None = None


@app.get("/api/health")
def health():
    count = os_client.count(index=INDEX_NAME)["count"]
    return {
        "status": "ok",
        "doc_count": count,
        "default_model": DEFAULT_MODEL,
        "embedding_model": "all-MiniLM-L6-v2",
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    query_vec = embed_model.encode(req.question, normalize_embeddings=True).tolist()
    if req.search_mode == "hybrid":
        hits = hybrid_search(os_client, req.question, query_vec, k=req.top_k)
    else:
        hits = knn_search(os_client, query_vec, k=req.top_k)

    context = build_context(hits)
    user_message = (
        f"Context from Vector Podcast transcripts:\n\n{context}\n\n"
        f"---\nQuestion: {req.question}"
    )
    model = req.model or DEFAULT_MODEL

    def event_stream():
        yield json.dumps({"type": "sources", "sources": hits}) + "\n"
        try:
            stream = bifrost_client.chat.completions.create(
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
                    yield json.dumps({"type": "token", "content": delta}) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# ── Diagnostics: triage retrieval failures before tuning anything ───────────


@app.post("/api/diagnose/irrelevant-documents")
def diagnose_irrelevant_documents_endpoint(req: DiagnoseIrrelevantRequest):
    """Step 1: is the answer even in the corpus / retrieved set?"""
    report = diagnose_irrelevant_documents(
        os_client, embed_model, req.query, req.answer_substring, k=req.top_k
    )
    return {
        "query": report.query,
        "vector_hits": report.vector_hits,
        "bm25_hits": report.bm25_hits,
        "vector_found_answer": report.vector_found_answer,
        "bm25_found_answer": report.bm25_found_answer,
        "diagnosis": report.diagnosis,
    }


@app.post("/api/diagnose/answer-position")
def diagnose_answer_position_endpoint(req: DiagnoseAnswerPositionRequest):
    """Step 2: right documents, wrong answer -- is it lost in the middle?"""
    report = diagnose_answer_position(
        os_client, embed_model, req.query, req.answer_substring, k=req.top_k
    )
    return {
        "query": report.query,
        "answer_rank": report.answer_rank,
        "reranked_answer_rank": report.reranked_answer_rank,
        "diagnosis": report.diagnosis,
        "hits": report.hits,
        "reranked_hits": report.reranked_hits,
    }


@app.post("/api/diagnose/phrasing-sensitivity")
def diagnose_phrasing_sensitivity_endpoint(req: DiagnosePhrasingRequest):
    """Step 3: right sometimes, wrong sometimes -- is it query formulation?"""
    report = diagnose_phrasing_sensitivity(os_client, embed_model, req.variants, k=req.top_k)
    return {
        "base_query": report.base_query,
        "variants": report.variants,
        "overlap_ratio": report.overlap_ratio,
        "diagnosis": report.diagnosis,
        "hit_sets": report.hit_sets,
    }


@app.post("/api/diagnose/latency")
def diagnose_latency_endpoint(req: DiagnoseLatencyRequest):
    """Step 4: everything is slow -- which stage is the bottleneck?"""
    report = diagnose_latency(
        os_client, embed_model, req.query, ask, k=req.top_k, use_reranking=req.use_reranking
    )
    return {
        "query": report.query,
        "stage_ms": report.stage_ms,
        "bottleneck": report.bottleneck,
        "diagnosis": report.diagnosis,
    }


# ── RAGAS: pipeline-wide quality evaluation ──────────────────────────────────


@app.get("/api/eval-set")
def get_eval_set():
    """The default labeled question set, so the UI can display/edit/extend it."""
    return [{"question": e.question, "ground_truth": e.ground_truth} for e in EVAL_SET]


@app.post("/api/evaluate")
def evaluate_endpoint(req: EvaluateRequest):
    """Score the pipeline against a labeled question set (the default EVAL_SET,
    or a custom `examples` list from the UI), streaming progress as it goes
    (retrieval+generation per question, then RAGAS judging per metric). Takes
    a few minutes: each sample runs several judge-LLM calls (faithfulness,
    relevancy, context precision/recall).
    """
    events: queue.Queue = queue.Queue()
    examples = (
        [EvalExample(question=e.question, ground_truth=e.ground_truth) for e in req.examples]
        if req.examples
        else None
    )

    def on_event(record: dict):
        events.put({"type": "log", **record})

    def worker():
        try:
            result = run_evaluation_json(
                top_k=req.top_k,
                client=os_client,
                embed_model=embed_model,
                on_event=on_event,
                examples=examples,
            )
            events.put({"type": "result", **result})
        except Exception as exc:
            events.put({"type": "error", "message": str(exc)})
        finally:
            events.put({"type": "done"})

    threading.Thread(target=worker, daemon=True).start()

    def event_stream():
        while True:
            event = events.get()
            yield json.dumps(event) + "\n"
            if event["type"] == "done":
                break

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
