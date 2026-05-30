"""Unit tests for pipeline stages, flow, and workers."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import QualityGateError
from src.ingestion.schemas import NoteRecord, NoteType
from src.pipeline.schemas import Entity, PipelineContext, QualityGateResult, Segment
from src.pipeline.stages.quality_gate import _build_expectations, quality_gate
from src.pipeline.stages.segmenter import _segment_sync

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_note(
    text: str = "Patient presents with chest pain.", note_id: str = "note-1"
) -> NoteRecord:
    return NoteRecord(
        note_id=note_id,
        patient_id="patient-1",
        note_type=NoteType.PROGRESS_NOTE,
        authored_at=datetime.now(tz=UTC),
        raw_text=text,
        source="fhir",
    )


def _make_ctx(text: str = "Patient presents with chest pain.") -> PipelineContext:
    note = _make_note(text)
    ctx = PipelineContext(note=note)
    ctx.deidentified_text = text
    ctx.segments = [Segment(text=text, start_char=0, end_char=len(text), sentence_index=0)]
    return ctx


# ── PipelineContext ───────────────────────────────────────────────────────────


class TestPipelineContext:
    def test_processed_text_prefers_deidentified(self) -> None:
        ctx = _make_ctx()
        ctx.deidentified_text = "DEIDENTIFIED text"
        assert ctx.processed_text == "DEIDENTIFIED text"

    def test_processed_text_falls_back_to_raw(self) -> None:
        ctx = PipelineContext(note=_make_note("raw note text"))
        assert ctx.processed_text == "raw note text"

    def test_errors_list_starts_empty(self) -> None:
        ctx = PipelineContext(note=_make_note())
        assert ctx.errors == []


# ── segmenter ─────────────────────────────────────────────────────────────────


class TestSegmenter:
    def test_segment_sync_splits_into_sentences(self) -> None:
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_sent1 = MagicMock()
        mock_sent1.text = "Patient has fever."
        mock_sent1.start_char = 0
        mock_sent1.end_char = 18
        mock_sent2 = MagicMock()
        mock_sent2.text = "BP is 120/80."
        mock_sent2.start_char = 19
        mock_sent2.end_char = 32
        mock_doc.sents = [mock_sent1, mock_sent2]
        mock_nlp.return_value = mock_doc

        with patch("src.pipeline.stages.segmenter._load_nlp", return_value=mock_nlp):
            segments = _segment_sync("Patient has fever. BP is 120/80.")

        assert len(segments) == 2
        assert segments[0].text == "Patient has fever."
        assert segments[1].sentence_index == 1

    def test_segment_sync_skips_empty_sentences(self) -> None:
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_sent = MagicMock()
        mock_sent.text = "   "  # whitespace only
        mock_sent.start_char = 0
        mock_sent.end_char = 3
        mock_doc.sents = [mock_sent]
        mock_nlp.return_value = mock_doc

        with patch("src.pipeline.stages.segmenter._load_nlp", return_value=mock_nlp):
            segments = _segment_sync("   ")

        assert len(segments) == 0

    async def test_segment_task_populates_ctx(self) -> None:
        from src.pipeline.stages.segmenter import segment

        ctx = PipelineContext(note=_make_note("Patient has fever. BP is 120/80."))
        expected_segments = [
            Segment("Patient has fever.", 0, 18, 0),
            Segment("BP is 120/80.", 19, 32, 1),
        ]

        with patch("src.pipeline.stages.segmenter._segment_sync", return_value=expected_segments):
            result = await segment(ctx)

        assert len(result.segments) == 2


# ── quality_gate ──────────────────────────────────────────────────────────────


class TestQualityGate:
    def test_all_expectations_pass_for_valid_note(self) -> None:
        ctx = _make_ctx("Patient presents with chest pain and shortness of breath.")
        expectations = _build_expectations(ctx)
        assert all(e["passed"] for e in expectations)

    def test_fails_when_text_too_short(self) -> None:
        ctx = _make_ctx("Short.")
        ctx.deidentified_text = "Short."
        expectations = _build_expectations(ctx)
        failed = [e for e in expectations if not e["passed"]]
        assert any("text length" in e["detail"] for e in failed)

    def test_fails_when_no_segments(self) -> None:
        ctx = _make_ctx()
        ctx.segments = []
        expectations = _build_expectations(ctx)
        failed = [e for e in expectations if not e["passed"]]
        assert any("segment" in e["detail"] for e in failed)

    async def test_quality_gate_task_passes_valid_note(self) -> None:
        ctx = _make_ctx("Patient presents with chest pain and shortness of breath today.")
        result = await quality_gate(ctx)
        assert result.quality is not None
        assert result.quality.passed

    async def test_quality_gate_task_raises_on_failure_when_strict(self) -> None:
        ctx = _make_ctx("Short.")
        ctx.deidentified_text = "Short."
        with pytest.raises(QualityGateError):
            await quality_gate(ctx, fail_on_error=True)

    async def test_quality_gate_task_does_not_raise_when_not_strict(self) -> None:
        ctx = _make_ctx("Short.")
        ctx.deidentified_text = "Short."
        result = await quality_gate(ctx, fail_on_error=False)
        assert result.quality is not None
        assert not result.quality.passed


# ── flow ──────────────────────────────────────────────────────────────────────


class TestPipelineFlow:
    async def test_process_note_runs_all_stages(self) -> None:
        from src.pipeline.flow import process_note

        note = _make_note("Patient with fever and elevated troponin. BP 150/90.")
        mock_segments = [Segment("Patient with fever.", 0, 18, 0)]
        mock_entities = [Entity("fever", "DISEASE", 13, 18, 1.0)]

        with (
            patch("src.pipeline.flow.segment", new_callable=AsyncMock) as mock_seg,
            patch("src.pipeline.flow.deidentify", new_callable=AsyncMock) as mock_deident,
            patch("src.pipeline.flow.extract_entities", new_callable=AsyncMock) as mock_ner,
            patch("src.pipeline.flow.quality_gate", new_callable=AsyncMock) as mock_qg,
            patch("src.pipeline.flow.vectorize", new_callable=AsyncMock) as mock_vec,
        ):

            def _add_segments(ctx: PipelineContext) -> PipelineContext:
                ctx.segments = mock_segments
                return ctx

            def _add_deident(ctx: PipelineContext) -> PipelineContext:
                ctx.deidentified_text = ctx.note.raw_text
                return ctx

            def _add_entities(ctx: PipelineContext) -> PipelineContext:
                ctx.entities = mock_entities
                return ctx

            def _add_quality(ctx: PipelineContext) -> PipelineContext:
                ctx.quality = QualityGateResult(passed=True, suite_name="test")
                return ctx

            def _add_vector(ctx: PipelineContext) -> PipelineContext:
                ctx.vector_indexed = True
                return ctx

            mock_seg.side_effect = _add_segments
            mock_deident.side_effect = _add_deident
            mock_ner.side_effect = _add_entities
            mock_qg.side_effect = _add_quality
            mock_vec.side_effect = _add_vector

            ctx = await process_note(note)

        assert ctx.vector_indexed is True
        assert len(ctx.entities) == 1
        assert ctx.quality is not None and ctx.quality.passed

    async def test_process_note_skips_vectorizer_in_dry_run(self) -> None:
        from src.pipeline.flow import process_note

        note = _make_note("Dry run test note with sufficient content.")
        mock_segments = [Segment("Dry run.", 0, 8, 0)]

        with (
            patch("src.pipeline.flow.segment", new_callable=AsyncMock) as mock_seg,
            patch("src.pipeline.flow.deidentify", new_callable=AsyncMock) as mock_deident,
            patch("src.pipeline.flow.extract_entities", new_callable=AsyncMock) as mock_ner,
            patch("src.pipeline.flow.quality_gate", new_callable=AsyncMock) as mock_qg,
            patch("src.pipeline.flow.vectorize", new_callable=AsyncMock) as mock_vec,
        ):
            for mock, fn in [
                (mock_seg, lambda ctx: setattr(ctx, "segments", mock_segments) or ctx),
                (
                    mock_deident,
                    lambda ctx: setattr(ctx, "deidentified_text", ctx.note.raw_text) or ctx,
                ),
                (mock_ner, lambda ctx: ctx),
                (
                    mock_qg,
                    lambda ctx: setattr(
                        ctx, "quality", QualityGateResult(passed=True, suite_name="t")
                    )
                    or ctx,
                ),
            ]:
                mock.side_effect = fn

            ctx = await process_note(note, skip_vectorizer=True)

        mock_vec.assert_not_awaited()
        assert ctx.vector_indexed is False
