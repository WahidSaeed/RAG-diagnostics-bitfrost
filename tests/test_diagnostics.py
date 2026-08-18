"""Unit tests for src/diagnostics.py.

These isolate the diagnosis *logic* (the branching that decides what to tell
the user) from OpenSearch and the embedding model, which are injected
dependencies — so every test here runs instantly with no live services.
"""
from unittest.mock import MagicMock, patch

import pytest

from src import diagnostics


def make_hit(episode_title="ep", chunk_index=0, chunk_text="", score=1.0):
    return {
        "score": score,
        "episode_title": episode_title,
        "chunk_text": chunk_text,
        "url": "https://example.com",
        "pub_date": "2024-01-01",
        "chunk_index": chunk_index,
    }


def make_embed_model():
    """A SentenceTransformer stand-in: .encode(...).tolist() -> list[float]."""
    model = MagicMock()
    model.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
    return model


# ── 1. diagnose_irrelevant_documents ─────────────────────────────────────────


class TestDiagnoseIrrelevantDocuments:
    def test_no_answer_substring_gives_manual_inspection_verdict(self):
        with patch.object(diagnostics, "knn_search", return_value=[make_hit(chunk_text="foo")]), \
             patch.object(diagnostics, "bm25_search", return_value=[make_hit(chunk_text="bar")]):
            report = diagnostics.diagnose_irrelevant_documents(
                client=MagicMock(), embed_model=make_embed_model(), query="q"
            )
        assert report.vector_found_answer is None
        assert report.bm25_found_answer is None
        assert "inspect vector_hits/bm25_hits manually" in report.diagnosis

    def test_vector_finds_answer_retrieval_is_fine(self):
        with patch.object(diagnostics, "knn_search", return_value=[make_hit(chunk_text="HNSW was invented by Yury Malkov")]), \
             patch.object(diagnostics, "bm25_search", return_value=[make_hit(chunk_text="unrelated")]):
            report = diagnostics.diagnose_irrelevant_documents(
                client=MagicMock(), embed_model=make_embed_model(),
                query="who invented HNSW", answer_substring="Yury Malkov",
            )
        assert report.vector_found_answer is True
        assert "Vector search finds it" in report.diagnosis

    def test_only_bm25_finds_answer_points_at_embeddings(self):
        with patch.object(diagnostics, "knn_search", return_value=[make_hit(chunk_text="unrelated")]), \
             patch.object(diagnostics, "bm25_search", return_value=[make_hit(chunk_text="Yury Malkov invented HNSW")]):
            report = diagnostics.diagnose_irrelevant_documents(
                client=MagicMock(), embed_model=make_embed_model(),
                query="who invented HNSW", answer_substring="Yury Malkov",
            )
        assert report.vector_found_answer is False
        assert report.bm25_found_answer is True
        assert "investigate embeddings" in report.diagnosis

    def test_neither_finds_answer_points_at_ingestion(self):
        with patch.object(diagnostics, "knn_search", return_value=[make_hit(chunk_text="unrelated")]), \
             patch.object(diagnostics, "bm25_search", return_value=[make_hit(chunk_text="also unrelated")]):
            report = diagnostics.diagnose_irrelevant_documents(
                client=MagicMock(), embed_model=make_embed_model(),
                query="who invented HNSW", answer_substring="Yury Malkov",
            )
        assert report.vector_found_answer is False
        assert report.bm25_found_answer is False
        assert "problem is ingestion" in report.diagnosis

    def test_answer_only_in_episode_title_still_counts_as_found(self):
        # Regression: a guest's name often lives only in the episode title
        # frontmatter, never spoken verbatim in the transcript body. Since
        # build_context() sends "Episode: {episode_title}" to the LLM, that's
        # still real grounding -- checking chunk_text alone would wrongly
        # diagnose this as an ingestion failure.
        hit = make_hit(
            episode_title="Yury Malkov - Staff Engineer, Twitter - Author of HNSW",
            chunk_text="yeah so there wasn't a lot of serendipity in inventing it",
        )
        with patch.object(diagnostics, "knn_search", return_value=[hit]), \
             patch.object(diagnostics, "bm25_search", return_value=[hit]):
            report = diagnostics.diagnose_irrelevant_documents(
                client=MagicMock(), embed_model=make_embed_model(),
                query="who invented HNSW", answer_substring="Yury Malkov",
            )
        assert report.vector_found_answer is True
        assert report.bm25_found_answer is True
        assert "Vector search finds it" in report.diagnosis

    def test_substring_match_is_case_insensitive(self):
        with patch.object(diagnostics, "knn_search", return_value=[make_hit(chunk_text="invented by YURY MALKOV")]), \
             patch.object(diagnostics, "bm25_search", return_value=[]):
            report = diagnostics.diagnose_irrelevant_documents(
                client=MagicMock(), embed_model=make_embed_model(),
                query="q", answer_substring="yury malkov",
            )
        assert report.vector_found_answer is True

    def test_both_find_it_takes_vector_branch(self):
        with patch.object(diagnostics, "knn_search", return_value=[make_hit(chunk_text="Yury Malkov")]), \
             patch.object(diagnostics, "bm25_search", return_value=[make_hit(chunk_text="Yury Malkov")]):
            report = diagnostics.diagnose_irrelevant_documents(
                client=MagicMock(), embed_model=make_embed_model(),
                query="q", answer_substring="Yury Malkov",
            )
        assert "Vector search finds it" in report.diagnosis


# ── 2. diagnose_answer_position ──────────────────────────────────────────────


class TestDiagnoseAnswerPosition:
    def test_answer_missing_entirely(self):
        hits = [make_hit(chunk_text="nothing relevant") for _ in range(5)]
        with patch.object(diagnostics, "hybrid_search", return_value=hits), \
             patch.object(diagnostics, "rerank", return_value=hits):
            report = diagnostics.diagnose_answer_position(
                client=MagicMock(), embed_model=make_embed_model(),
                query="q", answer_substring="Yury Malkov",
            )
        assert report.answer_rank is None
        assert "irrelevant_documents problem" in report.diagnosis

    def test_answer_already_near_top_position_is_fine(self):
        hits = [make_hit(chunk_text="Yury Malkov invented HNSW")] + [make_hit() for _ in range(4)]
        with patch.object(diagnostics, "hybrid_search", return_value=hits), \
             patch.object(diagnostics, "rerank", return_value=hits):
            report = diagnostics.diagnose_answer_position(
                client=MagicMock(), embed_model=make_embed_model(),
                query="q", answer_substring="Yury Malkov",
            )
        assert report.answer_rank == 1
        assert "already near the top" in report.diagnosis

    def test_rank_exactly_two_counts_as_near_top(self):
        hits = [make_hit(), make_hit(chunk_text="Yury Malkov")] + [make_hit() for _ in range(3)]
        with patch.object(diagnostics, "hybrid_search", return_value=hits), \
             patch.object(diagnostics, "rerank", return_value=hits):
            report = diagnostics.diagnose_answer_position(
                client=MagicMock(), embed_model=make_embed_model(),
                query="q", answer_substring="Yury Malkov",
            )
        assert report.answer_rank == 2
        assert "already near the top" in report.diagnosis

    def test_buried_but_reranking_promotes_it(self):
        hits = [make_hit() for _ in range(4)] + [make_hit(chunk_text="Yury Malkov")]  # rank 5
        reranked = [make_hit(chunk_text="Yury Malkov")] + [make_hit() for _ in range(4)]  # rank 1
        with patch.object(diagnostics, "hybrid_search", return_value=hits), \
             patch.object(diagnostics, "rerank", return_value=reranked):
            report = diagnostics.diagnose_answer_position(
                client=MagicMock(), embed_model=make_embed_model(),
                query="q", answer_substring="Yury Malkov",
            )
        assert report.answer_rank == 5
        assert report.reranked_answer_rank == 1
        assert "Add a reranking stage" in report.diagnosis

    def test_buried_and_reranking_does_not_help(self):
        hits = [make_hit() for _ in range(4)] + [make_hit(chunk_text="Yury Malkov")]  # rank 5
        with patch.object(diagnostics, "hybrid_search", return_value=hits), \
             patch.object(diagnostics, "rerank", return_value=hits):  # unchanged order
            report = diagnostics.diagnose_answer_position(
                client=MagicMock(), embed_model=make_embed_model(),
                query="q", answer_substring="Yury Malkov",
            )
        assert report.answer_rank == 5
        assert report.reranked_answer_rank == 5
        assert "fewer/shorter chunks" in report.diagnosis


# ── 3. diagnose_phrasing_sensitivity ─────────────────────────────────────────


class TestDiagnosePhrasingSensitivity:
    def test_requires_at_least_two_variants(self):
        with pytest.raises(ValueError):
            diagnostics.diagnose_phrasing_sensitivity(
                client=MagicMock(), embed_model=make_embed_model(), variants=["only one"]
            )

    def test_identical_hit_sets_are_stable(self):
        shared_hits = [make_hit(episode_title="ep1", chunk_index=i) for i in range(3)]
        with patch.object(diagnostics, "hybrid_search", return_value=shared_hits):
            report = diagnostics.diagnose_phrasing_sensitivity(
                client=MagicMock(), embed_model=make_embed_model(),
                variants=["what is HNSW", "explain HNSW"],
            )
        assert report.overlap_ratio == 1.0
        assert "stable across phrasings" in report.diagnosis

    def test_no_overlap_flags_query_formulation(self):
        hit_sets = [
            [make_hit(episode_title="ep1", chunk_index=0)],
            [make_hit(episode_title="ep2", chunk_index=1)],
        ]
        with patch.object(diagnostics, "hybrid_search", side_effect=hit_sets):
            report = diagnostics.diagnose_phrasing_sensitivity(
                client=MagicMock(), embed_model=make_embed_model(),
                variants=["what is HNSW", "totally different topic"],
            )
        assert report.overlap_ratio == 0.0
        assert "query formulation is likely the root cause" in report.diagnosis

    def test_partial_overlap_is_moderate_drift(self):
        # 2 shared out of 4 union -> overlap_ratio 0.5, lands in the [0.3, 0.7) band
        hit_sets = [
            [make_hit(episode_title="ep", chunk_index=i) for i in range(3)],  # {0,1,2}
            [make_hit(episode_title="ep", chunk_index=i) for i in (1, 2, 3)],  # {1,2,3}
        ]
        with patch.object(diagnostics, "hybrid_search", side_effect=hit_sets):
            report = diagnostics.diagnose_phrasing_sensitivity(
                client=MagicMock(), embed_model=make_embed_model(),
                variants=["what is HNSW", "explain HNSW briefly"],
            )
        assert report.overlap_ratio == pytest.approx(0.5)
        assert "Moderate drift" in report.diagnosis

    @pytest.mark.parametrize(
        "ratio_hits,expected_phrase",
        [
            (0.7, "stable across phrasings"),
            (0.3, "Moderate drift"),
        ],
    )
    def test_boundary_thresholds_are_inclusive(self, ratio_hits, expected_phrase):
        # Build hit sets whose overlap ratio lands exactly on the threshold.
        # union = shared + unique_a + unique_b; ratio = shared / union.
        if ratio_hits == 0.7:
            # 14 shared, 3 unique to each side -> 14 / 20 = 0.7
            shared = [make_hit(episode_title="ep", chunk_index=i) for i in range(14)]
            a = shared + [make_hit(episode_title="a", chunk_index=100 + i) for i in range(3)]
            b = shared + [make_hit(episode_title="b", chunk_index=200 + i) for i in range(3)]
        else:
            # 6 shared, 7 unique to each side -> 6 / 20 = 0.3
            shared = [make_hit(episode_title="ep", chunk_index=i) for i in range(6)]
            a = shared + [make_hit(episode_title="a", chunk_index=100 + i) for i in range(7)]
            b = shared + [make_hit(episode_title="b", chunk_index=200 + i) for i in range(7)]
        with patch.object(diagnostics, "hybrid_search", side_effect=[a, b]):
            report = diagnostics.diagnose_phrasing_sensitivity(
                client=MagicMock(), embed_model=make_embed_model(),
                variants=["q1", "q2"],
            )
        assert report.overlap_ratio == pytest.approx(ratio_hits)
        assert expected_phrase in report.diagnosis


# ── 4. diagnose_latency ──────────────────────────────────────────────────────


class TestDiagnoseLatency:
    def _run_with_timings(self, embedding_ms, retrieval_ms, generation_ms, reranking_ms=None):
        """Patch time.perf_counter with a scripted sequence so each stage's
        measured duration is exactly the value given, deterministically."""
        use_reranking = reranking_ms is not None
        deltas = [embedding_ms, retrieval_ms]
        if use_reranking:
            deltas.append(reranking_ms)
        deltas.append(generation_ms)

        ticks = []
        t = 0.0
        for d in deltas:
            ticks.append(t)             # t0 read, start of stage
            t += d / 1000.0
            ticks.append(t)             # end read, after stage work

        with patch.object(diagnostics, "hybrid_search", return_value=[make_hit()]), \
             patch.object(diagnostics, "rerank", return_value=[make_hit()]), \
             patch("time.perf_counter", side_effect=ticks):
            return diagnostics.diagnose_latency(
                client=MagicMock(), embed_model=make_embed_model(),
                query="q", ask_fn=MagicMock(return_value="answer"),
                use_reranking=use_reranking,
            )

    def test_reranking_stage_included_when_enabled(self):
        report = self._run_with_timings(
            embedding_ms=10, retrieval_ms=10, reranking_ms=10, generation_ms=10
        )
        assert set(report.stage_ms.keys()) == {"embedding", "retrieval", "reranking", "generation"}

    def test_reranking_stage_excluded_when_disabled(self):
        report = self._run_with_timings(embedding_ms=10, retrieval_ms=10, generation_ms=10)
        assert set(report.stage_ms.keys()) == {"embedding", "retrieval", "generation"}

    def test_bottleneck_is_the_largest_stage(self):
        report = self._run_with_timings(
            embedding_ms=5, retrieval_ms=5, reranking_ms=5, generation_ms=500
        )
        assert report.bottleneck == "generation"
        assert "'generation' is the largest slice" in report.diagnosis
        assert "not the vector database" in report.diagnosis

    def test_bottleneck_can_be_retrieval(self):
        report = self._run_with_timings(
            embedding_ms=5, retrieval_ms=500, reranking_ms=5, generation_ms=5
        )
        assert report.bottleneck == "retrieval"

    def test_ask_fn_is_called_with_query_and_hits(self):
        ask_fn = MagicMock(return_value="answer")
        hits = [make_hit()]
        with patch.object(diagnostics, "hybrid_search", return_value=hits), \
             patch.object(diagnostics, "rerank", return_value=hits):
            diagnostics.diagnose_latency(
                client=MagicMock(), embed_model=make_embed_model(),
                query="who invented HNSW", ask_fn=ask_fn, use_reranking=False,
            )
        ask_fn.assert_called_once_with("who invented HNSW", hits)
