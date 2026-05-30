"""Unit tests for RAG pipeline: retriever, reranker, context_builder,
answer_generator, guardrails, and prompt_templates."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import GuardrailViolationError
from src.core.schemas import CitationSource, QueryResponse
from src.ingestion.schemas import NoteType
from src.rag.answer_generator import generate, stream_generate
from src.rag.context_builder import (
    BuiltContext,
    _count_tokens,
    _extract_excerpt,
    _parse_date,
    _parse_note_type,
    build_context,
)
from src.rag.guardrails import apply_guardrails, scan_for_phi
from src.rag.prompt_templates import get_template, list_templates, render_user_message
from src.rag.reranker import _cache_key, _score_and_sort, rerank
from src.rag.retriever import retrieve, retrieve_multi
from src.vector_store.search import SearchResult

_NOW = datetime.now(tz=UTC)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_result(
    note_id: str = "note-1",
    score: float = 0.9,
    raw_text: str = "Patient presents with chest pain.",
    patient_id: str = "p-1",
    note_type: str = "progress_note",
    authored_at: str = "2024-01-01T00:00:00+00:00",
) -> SearchResult:
    return SearchResult(
        note_id=note_id,
        score=score,
        payload={
            "note_id": note_id,
            "raw_text": raw_text,
            "patient_id": patient_id,
            "note_type": note_type,
            "authored_at": authored_at,
        },
    )


# ── prompt_templates ──────────────────────────────────────────────────────────


class TestPromptTemplates:
    def test_get_clinical_qa_template(self) -> None:
        tmpl = get_template("clinical_qa", "v1")
        assert "system" in tmpl
        assert "user" in tmpl
        assert "{context}" in tmpl["user"]
        assert "{question}" in tmpl["user"]

    def test_get_icd_suggestion_template(self) -> None:
        tmpl = get_template("icd_suggestion", "v1")
        assert "{note_text}" in tmpl["user"]

    def test_missing_template_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            get_template("nonexistent_template", "v1")

    def test_render_user_message(self) -> None:
        msg = render_user_message(
            "clinical_qa",
            "v1",
            context="[1] chest pain",
            question="What is the diagnosis?",
        )
        assert "chest pain" in msg
        assert "What is the diagnosis?" in msg

    def test_list_templates_returns_names(self) -> None:
        names = list_templates()
        assert "clinical_qa" in names
        assert "icd_suggestion" in names


# ── retriever ─────────────────────────────────────────────────────────────────


class TestRetriever:
    async def test_retrieve_calls_embed_and_search(self) -> None:
        mock_embedder = AsyncMock()
        mock_embedder.embed_one = AsyncMock(return_value=[0.1] * 768)

        expected = [_make_result("note-1")]

        with patch(
            "src.rag.retriever.search_similar", new_callable=AsyncMock, return_value=expected
        ):
            results = await retrieve(
                "what is the diagnosis?",
                "notes_biomedbert_v1",
                embedder=mock_embedder,
            )

        assert len(results) == 1
        assert results[0].note_id == "note-1"
        mock_embedder.embed_one.assert_awaited_once_with("what is the diagnosis?")

    async def test_retrieve_multi_returns_merged(self) -> None:
        mock_embedder = AsyncMock()
        mock_embedder.embed_one = AsyncMock(return_value=[0.1] * 768)

        expected = [_make_result("note-1"), _make_result("note-2", score=0.8)]

        with patch(
            "src.rag.retriever.search_multi_collection",
            new_callable=AsyncMock,
            return_value=expected,
        ):
            results = await retrieve_multi(
                "medications",
                ["notes_biomedbert_v1", "notes_lora_mistral_v1"],
                embedder=mock_embedder,
            )

        assert len(results) == 2

    async def test_retrieve_multi_empty_collections(self) -> None:
        mock_embedder = AsyncMock()
        results = await retrieve_multi("query", [], embedder=mock_embedder)
        assert results == []


# ── reranker ──────────────────────────────────────────────────────────────────


class TestRerankerCacheKey:
    def test_deterministic(self) -> None:
        assert _cache_key("query", ["n1", "n2"]) == _cache_key("query", ["n1", "n2"])

    def test_order_independent(self) -> None:
        assert _cache_key("q", ["n2", "n1"]) == _cache_key("q", ["n1", "n2"])

    def test_starts_with_prefix(self) -> None:
        assert _cache_key("q", ["n1"]).startswith("rerank:")


class TestReranker:
    async def test_returns_empty_for_no_candidates(self) -> None:
        result = await rerank("query", [])
        assert result == []

    async def test_uses_cache_hit(self) -> None:
        import json

        candidate = _make_result("note-cache")
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(["note-cache"]).encode())

        result = await rerank("q", [candidate], redis_client=mock_redis)
        assert len(result) == 1
        assert result[0].note_id == "note-cache"

    async def test_scores_and_caches_on_miss(self) -> None:
        candidates = [_make_result("n1", score=0.9), _make_result("n2", score=0.5)]
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)

        with patch(
            "src.rag.reranker._score_and_sort",
            return_value=candidates,
        ):
            result = await rerank("q", candidates, top_k=2, redis_client=mock_redis)

        assert len(result) == 2
        mock_redis.set.assert_awaited_once()

    async def test_top_k_limits_results(self) -> None:
        candidates = [_make_result(f"n{i}", score=float(i)) for i in range(5)]
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)

        with patch("src.rag.reranker._score_and_sort", return_value=candidates):
            result = await rerank("q", candidates, top_k=2, redis_client=mock_redis)

        assert len(result) == 2

    def test_score_and_sort_uses_cross_encoder(self) -> None:
        import numpy as np

        candidates = [
            _make_result("n1", raw_text="chest pain"),
            _make_result("n2", raw_text="fever cough"),
        ]
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.3, 0.9])

        with patch("src.rag.reranker._get_cross_encoder", return_value=mock_model):
            result = _score_and_sort("chest pain", candidates)

        assert result[0].note_id == "n2"  # higher score first
        assert result[1].note_id == "n1"


# ── context_builder ───────────────────────────────────────────────────────────


class TestContextBuilder:
    def test_builds_context_from_results(self) -> None:
        results = [_make_result(f"note-{i}", score=0.9 - i * 0.1) for i in range(3)]
        ctx = build_context(results)

        assert "[1]" in ctx.context_text
        assert len(ctx.citations) == 3
        assert ctx.total_tokens > 0
        assert ctx.citation_map[1] == "note-0"

    def test_empty_results_returns_empty_context(self) -> None:
        ctx = build_context([])
        assert ctx.context_text == ""
        assert ctx.citations == []
        assert ctx.total_tokens == 0

    def test_token_budget_limits_notes(self) -> None:
        # Create results with large text that will exceed budget
        big_text = "A" * 5000
        results = [_make_result(f"note-{i}", raw_text=big_text) for i in range(10)]
        ctx = build_context(results, max_tokens=200)
        assert len(ctx.citations) < 10

    def test_citation_map_is_accurate(self) -> None:
        results = [_make_result("note-x"), _make_result("note-y")]
        ctx = build_context(results)
        assert ctx.citation_map[1] == "note-x"
        assert ctx.citation_map[2] == "note-y"

    def test_count_tokens_approximation(self) -> None:
        assert _count_tokens("hello world") > 0
        assert _count_tokens("A" * 400) == 100

    def test_parse_note_type_valid(self) -> None:
        assert _parse_note_type("progress_note") == NoteType.PROGRESS_NOTE

    def test_parse_note_type_invalid_falls_back(self) -> None:
        assert _parse_note_type("invalid_type") == NoteType.UNKNOWN

    def test_parse_date_iso_string(self) -> None:
        dt = _parse_date("2024-03-15T10:30:00Z")
        assert dt.year == 2024

    def test_parse_date_invalid_returns_now(self) -> None:
        dt = _parse_date("not-a-date")
        assert dt.year >= 2024

    def test_extract_excerpt_prefers_processed(self) -> None:
        result = SearchResult(
            note_id="n1",
            score=0.9,
            payload={"raw_text": "raw", "processed_text": "processed"},
        )
        assert _extract_excerpt(result) == "processed"

    def test_extract_excerpt_falls_back_to_raw(self) -> None:
        result = SearchResult(
            note_id="n1",
            score=0.9,
            payload={"raw_text": "raw text here"},
        )
        assert _extract_excerpt(result) == "raw text here"


# ── guardrails ────────────────────────────────────────────────────────────────


class TestGuardrails:
    def test_apply_guardrails_appends_disclaimer(self) -> None:
        with patch("src.rag.guardrails.scan_for_phi", return_value=[]):
            result = apply_guardrails("The patient is stable.")
        assert "clinical decision support" in result.lower()
        assert "The patient is stable." in result

    def test_strict_mode_raises_on_phi(self) -> None:
        with patch("src.rag.guardrails.scan_for_phi", return_value=["PERSON"]):
            with pytest.raises(GuardrailViolationError):
                apply_guardrails("John Smith is the patient.", strict=True)

    def test_non_strict_mode_allows_phi_with_warning(self) -> None:
        with patch("src.rag.guardrails.scan_for_phi", return_value=["PERSON"]):
            result = apply_guardrails("John Smith is the patient.", strict=False)
        assert result  # returned something

    def test_scan_for_phi_no_phi_in_clean_text(self) -> None:
        with patch("src.rag.guardrails._get_analyzer") as mock_analyzer:
            mock_engine = MagicMock()
            mock_engine.analyze.return_value = []
            mock_analyzer.return_value = mock_engine
            phi = scan_for_phi("The patient has elevated troponin levels.")
        assert phi == []


# ── answer_generator ──────────────────────────────────────────────────────────


class TestAnswerGenerator:
    def _make_context(self) -> BuiltContext:
        citation = CitationSource(
            note_id="note-1",
            patient_id="p-1",
            note_type=NoteType.PROGRESS_NOTE,
            authored_at=_NOW,
            excerpt="Patient has chest pain.",
            score=0.9,
        )
        return BuiltContext(
            context_text="[1] Patient has chest pain.",
            citations=[citation],
            total_tokens=10,
            citation_map={1: "note-1"},
        )

    async def test_generate_returns_query_response(self) -> None:
        ctx = self._make_context()

        with (
            patch("src.rag.answer_generator.complete", new_callable=AsyncMock) as mock_complete,
            patch("src.rag.answer_generator.apply_guardrails", side_effect=lambda x, **_: x),
        ):
            mock_complete.return_value = "The patient has chest pain."
            response = await generate("What is the diagnosis?", ctx)

        assert isinstance(response, QueryResponse)
        assert "chest pain" in response.answer
        assert response.latency_ms >= 0

    async def test_stream_generate_yields_chunks(self) -> None:
        ctx = self._make_context()

        async def _mock_stream(*_: object, **__: object) -> object:
            for chunk in ["Hello", " world"]:
                yield chunk

        with patch("src.rag.answer_generator.stream_complete", side_effect=_mock_stream):
            chunks = []
            async for chunk in stream_generate("query", ctx):
                chunks.append(chunk)

        assert chunks == ["Hello", " world"]
