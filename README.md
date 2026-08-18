# OpenSearch + RAG Workshop
### A semantic search & RAG system over Vector Podcast transcripts, with diagnostics and evaluation built in

---

## What you'll build

A fully working **Retrieval-Augmented Generation (RAG)** pipeline that lets you ask
natural-language questions and get answers grounded in content from 33 podcast
episode transcripts about vector search and AI — plus the tooling to actually
trust the answers: a failure-triage panel and a RAGAS-based quality evaluator.

```md
Your Question
     │
     ▼
[Embed with sentence-transformers]
     │
     ▼
[Hybrid search on OpenSearch]  ←  33 podcast episodes, chunked + embedded
     │
     ▼
[Generate answer via Bifrost → Groq]
     │
     ▼
Grounded Answer + Episode Citations
```

---

## Architecture

| Layer | Technology | Purpose |
|---|---|---|
| Vector store | OpenSearch 2.13 (k-NN plugin) | Store & search 384-dim embeddings |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Convert text → vectors |
| Index type | HNSW with cosine similarity | Approximate nearest-neighbour search |
| Reranking | `ms-marco-MiniLM-L-6-v2` (cross-encoder) | Higher-precision re-scoring before generation |
| LLM gateway | [Bifrost](https://getbifrost.ai) → Groq | Unified API, fallback chain, caching, governance |
| Backend | FastAPI | Retrieval + generation + diagnostics + evaluation endpoints |
| Frontend | Next.js | Chat, Diagnostics, and Evaluation tabs |
| Data | 33 Vector Podcast transcripts (Whisper) | Knowledge base |

---

## Prerequisites

- Docker + Docker Compose
- Python 3.13+ and [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Node.js + npm (for the web frontend)
- A [Groq](https://console.groq.com/) API key

---

## Setup (5 minutes)

### 1. Configure secrets

Create a `.env` file at the repo root:

```bash
GROQ_API_KEY=your-groq-key
BIFROST_ENCRYPTION_KEY=any-random-string
```

### 2. Start OpenSearch + Bifrost

```bash
docker compose up -d
```

Wait ~30 seconds, then verify: http://localhost:9200 (should show `{"status": "green" ...}`).
OpenSearch Dashboards: http://localhost:5601. Bifrost's observability dashboard: http://localhost:8080.

### 3. Install dependencies

```bash
uv sync
cd web && npm install && cd ..
```

### 4. Index the podcast transcripts

The `vector-podcast/` folder has 33 episode transcripts as markdown. Index them with:

```bash
uv run python -c "
from src.opensearch_client import get_client, create_index, bulk_index
from src.parser import load_podcast_chunks
from sentence_transformers import SentenceTransformer

chunks = load_podcast_chunks('vector-podcast')
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([c.chunk_text for c in chunks], normalize_embeddings=True).tolist()

client = get_client()
create_index(client)
bulk_index(client, chunks, embeddings)
"
```

### 5. Start the backend and frontend

```bash
# Backend (retrieval + generation + diagnostics + evaluation)
uv run uvicorn backend.main:app --reload --port 8000

# Frontend, in a separate terminal
cd web && npm run dev   # http://localhost:3000
```

Open http://localhost:3000 — you'll land on **Chat**, with **Diagnostics** and **Evaluation** tabs alongside it.

---

## Chat

A streaming chat interface with source citations and sidebar controls (search mode,
top-k, model). Generation is routed through **[Bifrost](https://getbifrost.ai)**, an
AI gateway sitting in front of Groq, rather than calling Groq's SDK directly.

The frontend expects the API at `NEXT_PUBLIC_API_URL` (see `web/.env.local`,
defaults to `http://localhost:8000`).

**Gateway features used** (config in `bifrost/data/config.json`):
- **Unified API** — `src/rag.py` talks to Bifrost's OpenAI-compatible
  `/v1/chat/completions` endpoint (`base_url=http://localhost:8080/v1`) using
  provider-prefixed model names.
- **Automatic fallback chain** — `groq/groq/compound` (Groq's agentic model,
  no separate daily token cap) → `groq/groq/compound-mini` →
  `groq/openai/gpt-oss-120b` → `groq/openai/gpt-oss-20b`, passed as
  `extra_body={"fallbacks": [...]}` so Bifrost retries the next model if one
  errors, times out, or is rate-limited.
- **Prompt/response caching** — a Redis-backed `semantic_cache` plugin
  (`dimension: 1` = direct/exact-hash mode) caches identical requests
  (`x-bf-cache-key: vector-podcast-rag`) for 24 hours, so repeated questions —
  including re-running the same Evaluation question set — skip the LLM call
  entirely. Verified via Bifrost's own request log (`cache_debug.cache_hit`).
- **Governance** — a virtual key (`vk-rag-app`) with a $20/day budget and a
  200k-token/500-request hourly rate limit, visible in the dashboard below.
- **Observability dashboard** — open http://localhost:8080 to see request
  logs, cache hit rate, latency, and spend in real time.

---

## Diagnostics

Before tuning the embedding model, chunk size, or prompt, diagnose *why* a query
is failing. `src/diagnostics.py` implements four checks, in the order you should
run them, and the web UI's **Diagnostics** tab exposes them as interactive forms:

| Check | Question it answers | How |
|---|---|---|
| Irrelevant documents | Is the answer in the corpus at all? | Compares vector (k-NN) vs. BM25 top-k for the same query — if neither finds it, the problem is ingestion, not retrieval |
| Answer position | Right documents, wrong answer? | Locates the answer's rank in the retrieved context and checks whether cross-encoder reranking rescues it (lost-in-the-middle detection) |
| Phrasing sensitivity | Right sometimes, wrong sometimes? | Runs paraphrased variants of a question and measures how much the retrieved set overlaps — low overlap points at query formulation |
| Latency breakdown | Everything is slow — where? | Times embedding, retrieval, reranking, and generation separately and names the actual bottleneck |

Served as FastAPI endpoints:

```
POST /api/diagnose/irrelevant-documents
POST /api/diagnose/answer-position
POST /api/diagnose/phrasing-sensitivity
POST /api/diagnose/latency
```

---

## Evaluation (RAGAS)

Where Diagnostics triages a single failing query, `src/evaluate.py` scores the
**whole pipeline** with [RAGAS](https://docs.ragas.io/) against a labeled
question set, so you can regression-test changes to chunk size, embedding
model, or reranking against a fixed baseline. The web UI's **Evaluation** tab:

- Shows the question set as **editable rows** (question + reference answer) —
  edit any of them, remove rows, or add your own with **+ Add question**.
- Streams live progress as the run happens: retrieval/generation per question,
  then RAGAS judging per metric, with a status dot per question (pending →
  in-progress → ✓ done / ⚠ error).
- Shows a live, color-coded telemetry console — one line per retrieval,
  generation, and individual metric judgment, with latency and score.
- Reports four metrics per question, plus their means: **faithfulness**
  (is the answer supported by the retrieved context?), **answer relevancy**,
  **context precision**, and **context recall**.

Every run also writes a full structured trace to `logs/evaluate-<run_id>.jsonl`
(one JSON record per event) — useful for `jq`-ing through a run after the fact,
independent of what the UI displayed live.

```
GET  /api/eval-set     # the default question set
POST /api/evaluate     # run evaluation (optionally with a custom `examples` list), streamed as NDJSON
```

```bash
# Or run it from the CLI:
uv run python -m src.evaluate
```

RAGAS's judge LLM uses the same Bifrost fallback chain as chat generation, so a
rate-limited primary model degrades gracefully instead of failing every metric.

---

## Project Structure

```md
OpenSearchVectorPodcastWorkshop/
├── README.md
├── docker-compose.yml              # OpenSearch + Dashboards + Bifrost + its cache
├── pyproject.toml
├── backend/
│   └── main.py                     # FastAPI service: chat, diagnostics, and evaluation endpoints
├── web/                            # Next.js chat + diagnostics + evaluation UI
│   ├── app/                        # App Router pages
│   ├── components/                 # Sidebar, chat view, diagnostics panel, evaluation panel
│   └── lib/                        # API client, shared types
├── bifrost/
│   └── data/
│       └── config.json             # Bifrost gateway config (providers, cache, governance)
├── vector-podcast/                 # 33 episode transcripts (markdown, Whisper-transcribed)
├── logs/                           # RAGAS evaluation traces (JSONL, git-ignored)
└── src/
    ├── parser.py                   # Markdown → PodcastChunk objects
    ├── opensearch_client.py        # Index management, search, Aiven client
    ├── rag.py                      # RAG pipeline, routed through Bifrost
    ├── reranker.py                 # Cross-encoder re-ranking
    ├── diagnostics.py              # RAG failure triage (irrelevant docs, answer position, phrasing, latency)
    └── evaluate.py                 # RAGAS pipeline evaluation + structured telemetry
```

---

## Key Concepts

### k-NN Index in OpenSearch

```json
{
  "settings": { "index.knn": true },
  "mappings": {
    "properties": {
      "embedding": {
        "type": "knn_vector",
        "dimension": 384,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "nmslib"
        }
      }
    }
  }
}
```

### HNSW: Hierarchical Navigable Small World

HNSW builds a multi-layer graph where each node connects to its nearest neighbours.
Search starts at the top (sparse) layer and zooms into the bottom (dense) layer,
pruning irrelevant branches early, achieving sub-millisecond search over millions of vectors.

Parameters:
- `ef_construction`: graph quality during indexing (higher = better recall, slower build)
- `m`: max connections per node (higher = better recall, more memory)
- `ef_search`: candidates explored at query time (higher = better recall, slower search)

### Chunking Strategy

Long podcast transcripts are split into overlapping windows:
- **chunk_size = 400 words** enough context for a coherent idea
- **overlap = 50 words** ensures ideas spanning chunk boundaries are captured

### RAG Prompt Design

The system prompt constrains the LLM to:
1. Answer only from provided context (no hallucination)
2. Cite the episode title for specific claims
3. Acknowledge when context is insufficient

---

## Sample Questions to Try

- *"What is HNSW and why did Yury Malkov invent it?"*
- *"How do wormhole vectors differ from standard hybrid search?"*
- *"What are the main challenges of running vector search in production?"*
- *"How does Pinecone's architecture differ from Weaviate's?"*
- *"What advice do guests give for evaluating embedding model quality?"*
- *"What is the role of sparse vectors in hybrid search?"*

---

## Teardown

```bash
docker compose down          # stop containers
docker compose down -v       # stop + delete the index data
```
