"""Unit tests for SQLAlchemy ORM models — instantiation and defaults."""

from datetime import UTC, datetime

from src.core.models import (
    ABTest,
    AuditLog,
    Base,
    DriftEvent,
    EmbeddingRecord,
    Note,
    PipelineJob,
    _utcnow,
)
from src.ingestion.schemas import NoteType


class TestUtcnow:
    def test_returns_utc_datetime(self) -> None:
        dt = _utcnow()
        assert dt.tzinfo is not None
        assert dt.tzinfo == UTC or dt.tzinfo.utcoffset(dt).total_seconds() == 0


class TestBase:
    def test_metadata_has_expected_tables(self) -> None:
        table_names = set(Base.metadata.tables.keys())
        assert "notes" in table_names
        assert "pipeline_jobs" in table_names
        assert "embedding_records" in table_names
        assert "ab_tests" in table_names
        assert "drift_events" in table_names
        assert "audit_logs" in table_names


class TestNoteModel:
    def test_table_name(self) -> None:
        assert Note.__tablename__ == "notes"

    def test_instantiation_with_required_fields(self) -> None:
        note = Note(
            patient_id="p-001",
            note_type=NoteType.PROGRESS_NOTE.value,
            authored_at=datetime.now(tz=UTC),
            raw_text="Patient presents with chest pain.",
            source="fhir",
            is_deidentified=False,
        )
        assert note.patient_id == "p-001"
        assert note.raw_text == "Patient presents with chest pain."
        assert note.source == "fhir"
        assert note.is_deidentified is False

    def test_indexes_defined(self) -> None:
        index_names = {idx.name for idx in Note.__table__.indexes}
        assert "ix_notes_patient_authored" in index_names
        assert "ix_notes_source_type" in index_names


class TestPipelineJobModel:
    def test_table_name(self) -> None:
        assert PipelineJob.__tablename__ == "pipeline_jobs"

    def test_instantiation(self) -> None:
        job = PipelineJob(note_id="note-001", status="pending")
        assert job.note_id == "note-001"
        assert job.status == "pending"


class TestEmbeddingRecordModel:
    def test_table_name(self) -> None:
        assert EmbeddingRecord.__tablename__ == "embedding_records"

    def test_instantiation(self) -> None:
        rec = EmbeddingRecord(
            note_id="note-001",
            model_name="biomedbert",
            model_version="v1",
            qdrant_collection="notes_v1",
            qdrant_point_id="point-abc",
        )
        assert rec.model_name == "biomedbert"
        assert rec.qdrant_collection == "notes_v1"


class TestABTestModel:
    def test_table_name(self) -> None:
        assert ABTest.__tablename__ == "ab_tests"

    def test_instantiation(self) -> None:
        ab = ABTest(
            name="biomedbert-vs-lora",
            model_a="biomedbert",
            model_b="lora-mistral",
            traffic_b_pct=0.1,
            is_active=True,
        )
        assert ab.traffic_b_pct == 0.1
        assert ab.is_active is True


class TestDriftEventModel:
    def test_table_name(self) -> None:
        assert DriftEvent.__tablename__ == "drift_events"

    def test_instantiation(self) -> None:
        event = DriftEvent(
            drift_type="embedding",
            metric_name="jensen_shannon_divergence",
            metric_value=0.15,
            threshold=0.10,
            alerted=False,
        )
        assert event.drift_type == "embedding"
        assert event.alerted is False


class TestAuditLogModel:
    def test_table_name(self) -> None:
        assert AuditLog.__tablename__ == "audit_logs"

    def test_instantiation(self) -> None:
        log = AuditLog(
            actor="user-42",
            action="read",
            resource_type="note",
            resource_id="note-001",
        )
        assert log.actor == "user-42"
        assert log.action == "read"

    def test_indexes_defined(self) -> None:
        index_names = {idx.name for idx in AuditLog.__table__.indexes}
        assert "ix_audit_actor_created" in index_names
        assert "ix_audit_resource" in index_names
