import json
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import IngestionError
from src.ingestion.batch_trigger import (
    _is_supported,
    _iter_fhir_resources,
    _parse_gcs_event,
    handle_gcs_event,
)


class TestParseGcsEvent:
    def test_returns_bucket_and_name(self) -> None:
        bucket, name = _parse_gcs_event({"bucket": "my-bucket", "name": "notes/file.json"})
        assert bucket == "my-bucket"
        assert name == "notes/file.json"

    def test_raises_on_missing_bucket(self) -> None:
        with pytest.raises(IngestionError):
            _parse_gcs_event({"name": "file.json"})

    def test_raises_on_missing_name(self) -> None:
        with pytest.raises(IngestionError):
            _parse_gcs_event({"bucket": "my-bucket"})


class TestIsSupported:
    def test_json_is_supported(self) -> None:
        assert _is_supported("notes/batch.json") is True

    def test_ndjson_is_supported(self) -> None:
        assert _is_supported("notes/batch.ndjson") is True

    def test_txt_is_supported(self) -> None:
        assert _is_supported("notes/note.txt") is True

    def test_csv_is_not_supported(self) -> None:
        assert _is_supported("notes/data.csv") is False

    def test_case_insensitive(self) -> None:
        assert _is_supported("notes/REPORT.JSON") is True


class TestIterFhirResources:
    def test_parses_single_resource(self) -> None:
        resource = {"resourceType": "DocumentReference", "id": "r1"}
        content = json.dumps(resource).encode()
        result = _iter_fhir_resources(content, "note.json")
        assert len(result) == 1
        assert result[0]["id"] == "r1"

    def test_parses_fhir_bundle(self) -> None:
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {"resource": {"resourceType": "DocumentReference", "id": "r1"}},
                {"resource": {"resourceType": "DiagnosticReport", "id": "r2"}},
            ],
        }
        result = _iter_fhir_resources(json.dumps(bundle).encode(), "bundle.json")
        assert len(result) == 2

    def test_parses_ndjson(self) -> None:
        lines = [
            json.dumps({"resourceType": "DocumentReference", "id": "r1"}),
            json.dumps({"resourceType": "DiagnosticReport", "id": "r2"}),
            "",
        ]
        content = "\n".join(lines).encode()
        result = _iter_fhir_resources(content, "batch.ndjson")
        assert len(result) == 2

    def test_parses_list(self) -> None:
        resources = [
            {"resourceType": "DocumentReference", "id": "r1"},
            {"resourceType": "DocumentReference", "id": "r2"},
        ]
        result = _iter_fhir_resources(json.dumps(resources).encode(), "list.json")
        assert len(result) == 2

    def test_raises_on_invalid_json(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _iter_fhir_resources(b"not json", "bad.json")


class TestHandleGcsEvent:
    def test_skips_unsupported_extension(self) -> None:
        result = handle_gcs_event({"bucket": "b", "name": "file.csv"})
        assert result["status"] == "skipped"

    def test_returns_error_on_parse_failure(self) -> None:
        with patch("src.ingestion.batch_trigger._read_gcs_object", return_value=b"bad json"):
            result = handle_gcs_event({"bucket": "b", "name": "file.json"})
        assert result["status"] == "error"

    def test_publishes_valid_file(self) -> None:
        resource = {
            "resourceType": "DocumentReference",
            "id": "r1",
            "subject": {"reference": "Patient/p1"},
        }
        content = json.dumps(resource).encode()

        mock_future = MagicMock()
        mock_future.result.return_value = "ok"
        mock_publisher = MagicMock()
        mock_publisher.topic_path.return_value = "projects/p/topics/notes"
        mock_publisher.publish.return_value = mock_future

        publisher_path = "src.ingestion.batch_trigger.pubsub_v1.PublisherClient"
        with (
            patch("src.ingestion.batch_trigger._read_gcs_object", return_value=content),
            patch(publisher_path, return_value=mock_publisher),
        ):
            result = handle_gcs_event({"bucket": "b", "name": "note.json"})

        assert result["status"] == "ok"
        assert result["total"] == 1
        assert result["published"] == 1
