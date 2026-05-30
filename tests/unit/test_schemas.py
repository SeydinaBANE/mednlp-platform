"""Unit tests for Pydantic v2 API schemas."""

from datetime import UTC, datetime

import pytest

from src.core.schemas import (
    AuditLogResponse,
    CitationSource,
    HealthResponse,
    IngestResponse,
    NoteDetailResponse,
    NoteResponse,
    PaginatedResponse,
    PipelineJobResponse,
    QueryRequest,
    QueryResponse,
    TokenRequest,
    TokenResponse,
)
from src.ingestion.schemas import NoteType

_NOW = datetime.now(tz=UTC)


class TestNoteResponse:
    def test_valid_instantiation(self) -> None:
        resp = NoteResponse(
            id="note-001",
            patient_id="p-001",
            encounter_id="enc-001",
            note_type=NoteType.PROGRESS_NOTE,
            authored_at=_NOW,
            source="fhir",
            is_deidentified=False,
            created_at=_NOW,
        )
        assert resp.id == "note-001"
        assert resp.note_type == NoteType.PROGRESS_NOTE

    def test_encounter_id_optional(self) -> None:
        resp = NoteResponse(
            id="note-002",
            patient_id="p-001",
            encounter_id=None,
            note_type=NoteType.UNKNOWN,
            authored_at=_NOW,
            source="api",
            is_deidentified=True,
            created_at=_NOW,
        )
        assert resp.encounter_id is None


class TestNoteDetailResponse:
    def test_includes_text_fields(self) -> None:
        resp = NoteDetailResponse(
            id="note-003",
            patient_id="p-001",
            encounter_id=None,
            note_type=NoteType.RADIOLOGY_REPORT,
            authored_at=_NOW,
            source="fhir",
            is_deidentified=True,
            created_at=_NOW,
            raw_text="Chest X-ray normal.",
            processed_text="chest xray normal",
            metadata={"fhir_resource_type": "DiagnosticReport"},
        )
        assert resp.raw_text == "Chest X-ray normal."
        assert resp.metadata["fhir_resource_type"] == "DiagnosticReport"


class TestQueryRequest:
    def test_valid_query(self) -> None:
        req = QueryRequest(query="What medications is the patient on?")
        assert req.top_k == 5
        assert req.stream is False

    def test_strips_whitespace(self) -> None:
        req = QueryRequest(query="  What is the diagnosis?  ")
        assert req.query == "What is the diagnosis?"

    def test_query_too_short_raises(self) -> None:
        with pytest.raises(ValueError):
            QueryRequest(query="Hi")

    def test_custom_top_k(self) -> None:
        req = QueryRequest(query="What is the prognosis?", top_k=10)
        assert req.top_k == 10

    def test_top_k_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            QueryRequest(query="What is the prognosis?", top_k=25)


class TestQueryResponse:
    def test_valid_instantiation(self) -> None:
        resp = QueryResponse(
            answer="The patient is on metformin.",
            sources=[
                CitationSource(
                    note_id="n1",
                    patient_id="p1",
                    note_type=NoteType.PROGRESS_NOTE,
                    authored_at=_NOW,
                    excerpt="metformin 1000mg BID",
                    score=0.92,
                )
            ],
            model="anthropic/claude-3.5-sonnet",
            latency_ms=450,
        )
        assert len(resp.sources) == 1
        assert resp.sources[0].score == 0.92


class TestPipelineJobResponse:
    def test_valid_instantiation(self) -> None:
        resp = PipelineJobResponse(
            id=1,
            note_id="note-001",
            status="success",
            stage="vectorizer",
            error=None,
            started_at=_NOW,
            finished_at=_NOW,
            duration_ms=320,
            created_at=_NOW,
        )
        assert resp.status == "success"
        assert resp.duration_ms == 320


class TestAuthSchemas:
    def test_token_request(self) -> None:
        req = TokenRequest(username="admin", password="secret")
        assert req.username == "admin"

    def test_token_response_defaults(self) -> None:
        resp = TokenResponse(access_token="jwt-token", expires_in=3600)
        assert resp.token_type == "bearer"


class TestMiscSchemas:
    def test_paginated_response(self) -> None:
        resp = PaginatedResponse(total=100, page=1, page_size=20, items=[1, 2, 3])
        assert resp.total == 100

    def test_ingest_response(self) -> None:
        resp = IngestResponse(note_id="n1", job_id=42, status="pending")
        assert resp.job_id == 42

    def test_health_response(self) -> None:
        resp = HealthResponse(status="ok", version="0.1.0", db="ok", redis="ok", qdrant="ok")
        assert resp.status == "ok"

    def test_audit_log_response(self) -> None:
        resp = AuditLogResponse(
            id=1,
            actor="user-1",
            action="read",
            resource_type="note",
            resource_id="n1",
            ip_address="127.0.0.1",
            details={},
            created_at=_NOW,
        )
        assert resp.actor == "user-1"
