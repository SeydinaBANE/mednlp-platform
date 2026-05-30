import pytest


@pytest.fixture(autouse=True, scope="session")
def prefect_test_harness() -> object:
    """Use Prefect's in-process test harness to avoid spawning a real server."""
    try:
        from prefect.testing.utilities import prefect_test_harness as _harness

        with _harness():
            yield
    except ImportError:
        yield


@pytest.fixture
def sample_fhir_document_reference() -> dict:
    return {
        "resourceType": "DocumentReference",
        "id": "note-001",
        "status": "current",
        "subject": {"reference": "Patient/patient-42"},
        "date": "2024-03-15T10:30:00Z",
        "context": {"encounter": [{"reference": "Encounter/enc-99"}]},
        "type": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "11506-3",
                    "display": "Progress note",
                }
            ]
        },
        "text": {
            "status": "generated",
            "div": (
                "<div xmlns='http://www.w3.org/1999/xhtml'>"
                "Patient presents with chest pain and shortness of breath. "
                "BP 140/90. HR 88. No acute distress. "
                "Assessment: Rule out ACS. Plan: EKG, troponin series, cardiology consult."
                "</div>"
            ),
        },
    }


@pytest.fixture
def sample_fhir_diagnostic_report() -> dict:
    return {
        "resourceType": "DiagnosticReport",
        "id": "report-002",
        "status": "final",
        "subject": {"reference": "Patient/patient-42"},
        "effectiveDateTime": "2024-03-15T12:00:00Z",
        "text": {
            "status": "generated",
            "div": "<div>Chest X-ray: No acute cardiopulmonary process.</div>",
        },
        "conclusion": "Normal chest radiograph.",
    }


@pytest.fixture
def sample_fhir_no_subject() -> dict:
    return {
        "resourceType": "DocumentReference",
        "id": "note-bad",
        "status": "current",
        "text": {
            "status": "generated",
            "div": "<div>Some note text.</div>",
        },
    }
