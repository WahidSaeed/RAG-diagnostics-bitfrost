"""OpenSearch index management and bulk ingestion."""
import os
from opensearchpy import OpenSearch, helpers
from .parser import PodcastChunk

INDEX_NAME = "vector-podcast"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


def get_client(host: str | None = None, port: int | None = None) -> OpenSearch:
    host = host or os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(port or os.getenv("OPENSEARCH_PORT", 9200))
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        timeout=30,
    )


INDEX_SETTINGS = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 100,
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }
    },
    "mappings": {
        "properties": {
            "embedding": {
                "type": "knn_vector",
                "dimension": EMBEDDING_DIM,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {"ef_construction": 128, "m": 24},
                },
            },
            "episode_title": {"type": "keyword"},
            "chunk_text": {"type": "text", "analyzer": "english"},
            "url": {"type": "keyword"},
            "pub_date": {"type": "keyword"},
            "description": {"type": "text"},
            "chunk_index": {"type": "integer"},
            "total_chunks": {"type": "integer"},
        }
    },
}


def create_index(client: OpenSearch, recreate: bool = False) -> None:
    exists = client.indices.exists(INDEX_NAME)
    if exists and recreate:
        client.indices.delete(INDEX_NAME)
        exists = False
    if not exists:
        client.indices.create(INDEX_NAME, body=INDEX_SETTINGS)
        print(f"Created index '{INDEX_NAME}'")
    else:
        print(f"Index '{INDEX_NAME}' already exists, skipping creation")


def bulk_index(
    client: OpenSearch,
    chunks: list[PodcastChunk],
    embeddings: list[list[float]],
    batch_size: int = 100,
) -> int:
    """Bulk-index chunks with their embeddings. Returns total docs indexed."""
    assert len(chunks) == len(embeddings), "chunks and embeddings must be same length"

    def _actions():
        for chunk, emb in zip(chunks, embeddings):
            yield {
                "_index": INDEX_NAME,
                "_id": chunk.doc_id,
                "_source": {
                    "embedding": emb,
                    "episode_title": chunk.episode_title,
                    "chunk_text": chunk.chunk_text,
                    "url": chunk.url,
                    "pub_date": chunk.pub_date,
                    "description": chunk.description,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                },
            }

    success, _ = helpers.bulk(client, _actions(), chunk_size=batch_size, request_timeout=60)
    return success


def knn_search(
    client: OpenSearch,
    query_embedding: list[float],
    k: int = 5,
    filters: dict | None = None,
) -> list[dict]:
    """Return top-k nearest-neighbour results."""
    knn_query: dict = {"vector": query_embedding, "k": k}
    body: dict = {"size": k, "query": {"knn": {"embedding": knn_query}}}

    if filters:
        body["query"] = {
            "bool": {
                "must": [{"knn": {"embedding": knn_query}}],
                "filter": [{"term": f} for f in filters.items()],
            }
        }

    response = client.search(index=INDEX_NAME, body=body)
    return [
        {
            "score": hit["_score"],
            "episode_title": hit["_source"]["episode_title"],
            "chunk_text": hit["_source"]["chunk_text"],
            "url": hit["_source"]["url"],
            "pub_date": hit["_source"]["pub_date"],
            "chunk_index": hit["_source"]["chunk_index"],
        }
        for hit in response["hits"]["hits"]
    ]


def bm25_search(
    client: OpenSearch,
    query_text: str,
    k: int = 5,
) -> list[dict]:
    """Pure lexical (BM25) search, no vector component.

    Used to check whether the answer exists in the corpus at all when
    vector search comes up empty (see diagnostics.py).
    """
    body = {
        "size": k,
        "query": {"match": {"chunk_text": {"query": query_text}}},
    }
    response = client.search(index=INDEX_NAME, body=body)
    return [
        {
            "score": hit["_score"],
            "episode_title": hit["_source"]["episode_title"],
            "chunk_text": hit["_source"]["chunk_text"],
            "url": hit["_source"]["url"],
            "pub_date": hit["_source"]["pub_date"],
            "chunk_index": hit["_source"]["chunk_index"],
        }
        for hit in response["hits"]["hits"]
    ]


def hybrid_search(
    client: OpenSearch,
    query_text: str,
    query_embedding: list[float],
    k: int = 5,
) -> list[dict]:
    """Combine BM25 lexical + k-NN vector search results."""
    body = {
        "size": k,
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_embedding,
                                "k": k,
                            }
                        }
                    },
                    {
                        "match": {
                            "chunk_text": {
                                "query": query_text,
                                "boost": 0.3,
                            }
                        }
                    },
                ]
            }
        },
    }
    response = client.search(index=INDEX_NAME, body=body)
    return [
        {
            "score": hit["_score"],
            "episode_title": hit["_source"]["episode_title"],
            "chunk_text": hit["_source"]["chunk_text"],
            "url": hit["_source"]["url"],
            "pub_date": hit["_source"]["pub_date"],
            "chunk_index": hit["_source"]["chunk_index"],
        }
        for hit in response["hits"]["hits"]
    ]


# ── Helpers for advanced notebooks ───────────────────────────────────────────


def knn_search_filtered(
    client: OpenSearch,
    query_embedding: list[float],
    k: int = 5,
    episode_title: str | None = None,
    pub_date_gte: str | None = None,
    pub_date_lte: str | None = None,
    index_name: str = INDEX_NAME,
) -> list[dict]:
    """k-NN search with optional metadata filters (post-filter)."""
    knn_query: dict = {"vector": query_embedding, "k": k}
    body: dict = {
        "size": k,
        "query": {"knn": {"embedding": knn_query}},
    }

    filters: list[dict] = []
    if episode_title:
        filters.append({"term": {"episode_title": episode_title}})
    if pub_date_gte or pub_date_lte:
        date_range: dict = {}
        if pub_date_gte:
            date_range["gte"] = pub_date_gte
        if pub_date_lte:
            date_range["lte"] = pub_date_lte
        filters.append({"range": {"pub_date": date_range}})

    if filters:
        body["post_filter"] = {"bool": {"must": filters}}

    response = client.search(index=index_name, body=body)
    return [
        {
            "score": hit["_score"],
            "episode_title": hit["_source"]["episode_title"],
            "chunk_text": hit["_source"]["chunk_text"],
            "url": hit["_source"]["url"],
            "pub_date": hit["_source"]["pub_date"],
            "chunk_index": hit["_source"]["chunk_index"],
        }
        for hit in response["hits"]["hits"]
    ]


def create_index_custom(
    client: OpenSearch,
    index_name: str,
    embedding_dim: int,
    recreate: bool = False,
) -> None:
    """Create a k-NN index with a custom embedding dimension."""
    exists = client.indices.exists(index_name)
    if exists and recreate:
        client.indices.delete(index_name)
        exists = False
    if not exists:
        settings = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {"ef_construction": 128, "m": 24},
                        },
                    },
                    "episode_title": {"type": "keyword"},
                    "chunk_text": {"type": "text", "analyzer": "english"},
                    "url": {"type": "keyword"},
                    "pub_date": {"type": "keyword"},
                    "description": {"type": "text"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                }
            },
        }
        client.indices.create(index_name, body=settings)
        print(f"Created index '{index_name}' with {embedding_dim}-dim embeddings")
    else:
        print(f"Index '{index_name}' already exists, skipping creation")


def get_aiven_client(
    host: str,
    port: int = 9200,
    username: str = "avnadmin",
    password: str = "",
) -> OpenSearch:
    """Connect to an Aiven-managed OpenSearch cluster over TLS."""
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(username, password),
        use_ssl=True,
        verify_certs=True,
        http_compress=True,
        timeout=30,
    )
