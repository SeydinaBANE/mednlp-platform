import base64

import pytest

from src.core.exceptions import FHIRParseError, MissingPatientReferenceError
from src.ingestion.fhir_parser import parse_fhir_resource, safe_get
from src.ingestion.schemas import NoteType


class TestSafeGet:
    def test_returns_nested_value(self) -> None:
        data = {"a": {"b": {"c": 42}}}
        assert safe_get(data, "a", "b", "c") == 42

    def test_returns_default_on_missing_key(self) -> None:
        data = {"a": {}}
        assert safe_get(data, "a", "b", "c", default="x") == "x"

    def test_returns_default_on_none_intermediate(self) -> None:
        data = {"a": None}
        assert safe_get(data, "a", "b", default=0) == 0

    def test_returns_none_default_by_default(self) -> None:
        assert safe_get({}, "missing") is None


class TestParseDocumentReference:
    def test_parses_valid_resource(self, sample_fhir_document_reference: dict) -> None:
        note = parse_fhir_resource(sample_fhir_document_reference)

        assert note.note_id == "note-001"
        assert note.patient_id == "patient-42"
        assert note.encounter_id == "enc-99"
        assert note.note_type == NoteType.PROGRESS_NOTE
        assert note.source == "fhir"
        assert "chest pain" in note.raw_text

    def test_strips_xhtml_tags_from_text(self, sample_fhir_document_reference: dict) -> None:
        note = parse_fhir_resource(sample_fhir_document_reference)
        assert "<div>" not in note.raw_text
        assert "</" not in note.raw_text

    def test_raises_on_missing_subject(self, sample_fhir_no_subject: dict) -> None:
        with pytest.raises(MissingPatientReferenceError) as exc_info:
            parse_fhir_resource(sample_fhir_no_subject)
        assert exc_info.value.resource_id == "note-bad"

    def test_raises_on_empty_text(self) -> None:
        resource = {
            "resourceType": "DocumentReference",
            "id": "note-empty",
            "subject": {"reference": "Patient/p1"},
            "date": "2024-01-01T00:00:00Z",
            "text": {"status": "empty", "div": ""},
        }
        with pytest.raises(FHIRParseError):
            parse_fhir_resource(resource)

    def test_uses_fallback_date_on_invalid(self, sample_fhir_document_reference: dict) -> None:
        resource = dict(sample_fhir_document_reference)
        resource["date"] = "not-a-date"
        # Should not raise, use current time fallback
        note = parse_fhir_resource(resource)
        assert note.authored_at is not None

    def test_unknown_note_type_on_unmapped_loinc(self) -> None:
        resource = {
            "resourceType": "DocumentReference",
            "id": "note-x",
            "subject": {"reference": "Patient/p1"},
            "date": "2024-01-01T00:00:00Z",
            "type": {"coding": [{"system": "http://loinc.org", "code": "99999-9"}]},
            "text": {"status": "generated", "div": "<div>Some text.</div>"},
        }
        note = parse_fhir_resource(resource)
        assert note.note_type == NoteType.UNKNOWN


class TestParseDiagnosticReport:
    def test_parses_valid_report(self, sample_fhir_diagnostic_report: dict) -> None:
        note = parse_fhir_resource(sample_fhir_diagnostic_report)

        assert note.note_id == "report-002"
        assert note.patient_id == "patient-42"
        assert note.note_type == NoteType.RADIOLOGY_REPORT
        assert "Normal chest radiograph" in note.raw_text

    def test_combines_text_and_conclusion(self, sample_fhir_diagnostic_report: dict) -> None:
        note = parse_fhir_resource(sample_fhir_diagnostic_report)
        assert "cardiopulmonary" in note.raw_text
        assert "Normal chest" in note.raw_text

    def test_raises_on_unsupported_resource_type(self) -> None:
        resource = {"resourceType": "Observation", "id": "obs-1"}
        with pytest.raises(FHIRParseError) as exc_info:
            parse_fhir_resource(resource)
        assert "Unsupported resourceType" in str(exc_info.value)

    def test_raises_on_missing_subject(self) -> None:
        resource = {
            "resourceType": "DiagnosticReport",
            "id": "report-no-subject",
            "status": "final",
            "text": {"status": "generated", "div": "<div>Some findings.</div>"},
        }
        with pytest.raises(MissingPatientReferenceError):
            parse_fhir_resource(resource)

    def test_raises_on_empty_text(self) -> None:
        resource = {
            "resourceType": "DiagnosticReport",
            "id": "report-empty",
            "subject": {"reference": "Patient/p1"},
            "effectiveDateTime": "2024-01-01T00:00:00Z",
            "text": {"status": "empty", "div": ""},
        }
        with pytest.raises(FHIRParseError):
            parse_fhir_resource(resource)

    def test_uses_fallback_date_on_invalid(self, sample_fhir_diagnostic_report: dict) -> None:
        resource = dict(sample_fhir_diagnostic_report)
        resource.pop("effectiveDateTime", None)
        resource.pop("issued", None)
        resource["meta"] = {"lastUpdated": "not-a-date"}
        note = parse_fhir_resource(resource)
        assert note.authored_at is not None


class TestBinaryFallback:
    def test_extracts_text_from_binary_contained_resource(self) -> None:
        payload = base64.b64encode(b"Chest X-ray normal.").decode()
        resource = {
            "resourceType": "DocumentReference",
            "id": "note-binary",
            "subject": {"reference": "Patient/p1"},
            "date": "2024-01-01T00:00:00Z",
            "contained": [{"resourceType": "Binary", "id": "b1", "data": payload}],
        }
        note = parse_fhir_resource(resource)
        assert "Chest X-ray normal" in note.raw_text

    def test_skips_corrupt_binary_and_falls_back(self) -> None:
        resource = {
            "resourceType": "DocumentReference",
            "id": "note-corrupt",
            "subject": {"reference": "Patient/p1"},
            "date": "2024-01-01T00:00:00Z",
            "contained": [{"resourceType": "Binary", "id": "b2", "data": "!!!not-base64!!!"}],
            "text": {"status": "generated", "div": "<div>Fallback text.</div>"},
        }
        note = parse_fhir_resource(resource)
        assert "Fallback text" in note.raw_text
