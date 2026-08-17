# OpenSearch + RAG Workshop
### Build a semantic search & RAG system over Vector Podcast transcripts

---

## What you'll build

A fully working **Retrieval-Augmented Generation (RAG)** pipeline that lets you ask
natural-language questions and get answers grounded in content from 33 podcast
episode transcripts about vector search and AI.

```md
Your Question
     │
     ▼
[Embed with sentence-transformers]
     │
     ▼
[k-NN search on OpenSearch]  ←  33 podcast episodes, chunked + embedded
     │
     ▼
[Generate answer with Ollama]
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
| LLM | Ollama (local, e.g. `gemma4`) | Generate grounded answers |
| Data | 33 Vector Podcast transcripts (Whisper) | Knowledge base |

---

## Prerequisites

- Docker + Docker Compose
- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com/) installed and running locally

---

## Setup (5 minutes)

### 1. Start OpenSearch

```bash
cd workshop/
docker compose up -d
```

Wait ~30 seconds, then verify: http://localhost:9200
You should see `{"status": "green" ...}`.

OpenSearch Dashboards (optional UI): http://localhost:5601

### 2. Install Python dependencies

```bash
uv sync
```

### 3. Pull an Ollama model

```bash
ollama pull gemma4
```

You can use any Ollama model. Set `OLLAMA_MODEL` to override the default:
```bash
export OLLAMA_MODEL=gemma4    # default
export OLLAMA_HOST=http://localhost:11434  # default
```

### 4. Launch Jupyter

```bash
jupyter notebook notebooks/
```

---

## Workshop Flow

### Part 1: Index Podcasts `01_index_podcasts.ipynb` (~15 min)

| Step | What happens |
|---|---|
| Connect | Verify OpenSearch is running |
| Create index | k-NN index with HNSW + cosine similarity, 384-dim |
| Parse | Load 33 `.md` files, extract frontmatter + transcript body |
| Chunk | Split transcripts into ~400-word windows with 50-word overlap |
| Embed | Generate sentence embeddings with `all-MiniLM-L6-v2` |
| Index | Bulk-ingest all chunks into OpenSearch |
| Verify | Run a test k-NN query |

### Part 2: RAG Pipeline `02_rag_pipeline.ipynb` (~15 min)

| Step | What happens |
|---|---|
| Semantic search | k-NN query finds meaning, not just keywords |
| Hybrid search | BM25 + k-NN combined for better term coverage |
| RAG pipeline | Retrieve → format context → generate with Ollama |
| Inspect context | See exactly what goes into the LLM prompt |
| Explore | Ask your own questions! |

---

## Advanced Notebooks (go-further, pick any)

### 03: Cross-Encoder Re-Ranking `03_reranking.ipynb`

Two-stage retrieval: bi-encoder retrieves top-20 candidates, then a **cross-encoder**
(`ms-marco-MiniLM-L-6-v2`) re-scores them for much higher precision before sending to the LLM.
Side-by-side RAG comparison shows the impact on answer quality.

### 04: Metadata Filtering `04_metadata_filtering.ipynb`

Scope k-NN search by **episode title**, **date range**, or any structured field.
Covers OpenSearch post-filtering, pre-filtering trade-offs, and the over-fetch pattern.

### 05: Larger Embeddings `05_larger_embeddings.ipynb`

Swap `all-MiniLM-L6-v2` (384d) for `all-mpnet-base-v2` (768d) and compare:
search quality, encoding speed, and index size side by side on the same data.

### 06: Aiven Managed OpenSearch `06_aiven_opensearch.ipynb`

Connect to an **Aiven for OpenSearch** cloud cluster over TLS. Create the same index,
bulk-index podcast chunks, and run RAG queries against a fully managed service.

### 07: Next.js Chat UI

A streaming chat interface with source citations, sidebar controls (search mode,
top-k, model), backed by a FastAPI service that wraps retrieval + LLM generation.
Generation is routed through **[Bifrost](https://getbifrost.ai)**, an AI gateway
sitting in front of Groq, rather than calling Groq's SDK directly.

```bash
# 1. Start OpenSearch + Bifrost (gateway + its Redis-backed cache)
docker compose up -d opensearch opensearch-dashboards bifrost-cache bifrost

# 2. Start the API (retrieval + generation via Bifrost)
uv run uvicorn backend.main:app --reload --port 8000

# 3. Start the frontend
cd web/
npm install   # first time only
npm run dev   # opens http://localhost:3000
```

The frontend expects the API at `NEXT_PUBLIC_API_URL` (see `web/.env.local`,
defaults to `http://localhost:8000`). Set `GROQ_API_KEY` and
`BIFROST_ENCRYPTION_KEY` in `.env` at the repo root before starting the containers.

**Gateway features used** (config in `bifrost/data/config.json`):
- **Unified API** — `src/rag.py` talks to Bifrost's OpenAI-compatible
  `/v1/chat/completions` endpoint (`base_url=http://localhost:8080/v1`) using
  provider-prefixed model names, e.g. `groq/openai/gpt-oss-120b`.
- **Automatic fallback** — each request passes
  `extra_body={"fallbacks": ["groq/openai/gpt-oss-20b"]}`, so Bifrost retries
  against a second model if the primary one errors or times out.
- **Direct-mode response caching** — a Redis-backed `semantic_cache` plugin
  caches identical requests (`x-bf-cache-key: vector-podcast-rag`) for 10
  minutes, so repeated questions skip the LLM call entirely.
- **Governance** — a virtual key (`vk-rag-app`) with a $20/day budget and a
  200k-token/500-request hourly rate limit, visible in the dashboard below.
- **Observability dashboard** — open http://localhost:8080 to see request
  logs, cache hit rate, latency, and spend in real time.

---

## Project Structure

```md
workshop/
├── README.md
├── docker-compose.yml              # OpenSearch + Dashboards + Bifrost + its cache
├── requirements.txt
├── backend/
│   └── main.py                     # FastAPI service (retrieval + generation via Bifrost)
├── web/                            # Next.js chat UI
│   ├── app/                        # App Router pages
│   ├── components/                 # Sidebar, chat message view
│   └── lib/                        # API client, shared types
├── bifrost/
│   └── data/
│       └── config.json             # Bifrost gateway config (providers, cache, governance)
├── notebooks/
│   ├── 01_index_podcasts.ipynb     # Core: parse, embed, index
│   ├── 02_rag_pipeline.ipynb       # Core: search + RAG
│   ├── 03_reranking.ipynb          # Advanced: cross-encoder re-ranking
│   ├── 04_metadata_filtering.ipynb # Advanced: filter by episode/date
│   ├── 05_larger_embeddings.ipynb  # Advanced: 768-dim embeddings
│   └── 06_aiven_opensearch.ipynb   # Advanced: managed cloud cluster
└── src/
    ├── parser.py                   # Markdown → PodcastChunk objects
    ├── opensearch_client.py        # Index management, search, Aiven client
    ├── rag.py                      # RAG pipeline, routed through Bifrost
    └── reranker.py                 # Cross-encoder re-ranking
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
