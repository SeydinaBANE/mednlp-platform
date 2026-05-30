"""Parse HL7 FHIR R4 resources into NoteRecord.

Handles missing optional fields gracefully via safe_get().
Notes without a subject reference go to DLQ immediately.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

import structlog

from src.core.exceptions import FHIRParseError, MissingPatientReferenceError
from src.ingestion.schemas import NoteRecord, NoteType

logger = structlog.get_logger(__name__)

_NOTE_TYPE_MAP: dict[str, NoteType] = {
    "11488-4": NoteType.CONSULTATION,
    "18842-5": NoteType.DISCHARGE_SUMMARY,
    "18726-0": NoteType.RADIOLOGY_REPORT,
    "11504-8": NoteType.OPERATIVE_REPORT,
    "11506-3": NoteType.PROGRESS_NOTE,
}


def safe_get(obj: dict[str, Any], *keys: str | int, default: Any = None) -> Any:  # noqa: ANN401
    """Traverse nested dict/list without raising on missing keys."""
    current: Any = obj
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        elif isinstance(current, list) and isinstance(key, int):
            current = current[key] if len(current) > key else default
        else:
            return default
        if current is None:
            return default
    return current


def _extract_note_type(resource: dict[str, Any]) -> NoteType:
    codings: list[dict[str, Any]] = safe_get(resource, "type", "coding", default=[])
    for coding in codings:
        code = safe_get(coding, "code", default="")
        if code in _NOTE_TYPE_MAP:
            return _NOTE_TYPE_MAP[code]
    return NoteType.UNKNOWN


def _extract_text(resource: dict[str, Any]) -> str | None:
    """Try text.div first, then fall back to contained resources."""
    div = safe_get(resource, "text", "div")
    if div and isinstance(div, str) and len(div.strip()) > 0:
        # Strip basic XHTML tags from narrative div
        import re

        return re.sub(r"<[^>]+>", " ", div).strip()

    # Fallback: look in contained resources for Binary or Attachment
    contained: list[dict[str, Any]] = safe_get(resource, "contained", default=[])
    for item in contained:
        if item.get("resourceType") == "Binary":
            data = safe_get(item, "data")
            if data:
                import base64

                try:
                    return base64.b64decode(data).decode("utf-8", errors="replace")
                except Exception:  # noqa: BLE001
                    logger.debug("binary_decode_failed", item_id=item.get("id"))

    return None


def parse_document_reference(resource: dict[str, Any]) -> NoteRecord:
    """Parse a FHIR R4 DocumentReference into a NoteRecord."""
    resource_id = resource.get("id") or str(uuid.uuid4())

    # Subject (patient) is required — poison pill if missing
    subject_ref = safe_get(resource, "subject", "reference")
    if not subject_ref:
        raise MissingPatientReferenceError(resource_id)

    patient_id = str(subject_ref).split("/")[-1]

    # Extract note text
    text = _extract_text(resource)
    if not text:
        raise FHIRParseError(resource_id, "No usable text content found")

    # Authored date
    authored_str = safe_get(resource, "date") or safe_get(resource, "meta", "lastUpdated")
    try:
        authored_at = datetime.fromisoformat(str(authored_str).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        logger.warning("fhir_date_parse_failed", resource_id=resource_id, raw=authored_str)
        authored_at = datetime.now(tz=UTC)

    encounter_ref = safe_get(resource, "context", "encounter", 0, "reference")
    encounter_id = str(encounter_ref).split("/")[-1] if encounter_ref else None

    return NoteRecord(
        note_id=resource_id,
        patient_id=patient_id,
        encounter_id=encounter_id,
        note_type=_extract_note_type(resource),
        authored_at=authored_at,
        raw_text=text,
        source="fhir",
        metadata={"fhir_resource_type": "DocumentReference"},
    )


def parse_diagnostic_report(resource: dict[str, Any]) -> NoteRecord:
    """Parse a FHIR R4 DiagnosticReport into a NoteRecord."""
    resource_id = resource.get("id") or str(uuid.uuid4())

    subject_ref = safe_get(resource, "subject", "reference")
    if not subject_ref:
        raise MissingPatientReferenceError(resource_id)

    patient_id = str(subject_ref).split("/")[-1]

    text = _extract_text(resource)
    # DiagnosticReport may have conclusion field as additional text
    conclusion = safe_get(resource, "conclusion", default="")
    combined = " ".join(filter(None, [text, conclusion])).strip()

    if not combined:
        raise FHIRParseError(resource_id, "No usable text content found in DiagnosticReport")

    authored_str = (
        safe_get(resource, "effectiveDateTime")
        or safe_get(resource, "issued")
        or safe_get(resource, "meta", "lastUpdated")
    )
    try:
        authored_at = datetime.fromisoformat(str(authored_str).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        authored_at = datetime.now(tz=UTC)

    encounter_ref = safe_get(resource, "encounter", "reference")
    encounter_id = str(encounter_ref).split("/")[-1] if encounter_ref else None

    return NoteRecord(
        note_id=resource_id,
        patient_id=patient_id,
        encounter_id=encounter_id,
        note_type=NoteType.RADIOLOGY_REPORT,
        authored_at=authored_at,
        raw_text=combined,
        source="fhir",
        metadata={"fhir_resource_type": "DiagnosticReport"},
    )


def parse_fhir_resource(resource: dict[str, Any]) -> NoteRecord:
    """Dispatch to the correct parser based on FHIR resourceType."""
    resource_type = resource.get("resourceType", "")

    if resource_type == "DocumentReference":
        return parse_document_reference(resource)
    if resource_type == "DiagnosticReport":
        return parse_diagnostic_report(resource)

    raise FHIRParseError(
        resource.get("id", "unknown"),
        f"Unsupported resourceType: {resource_type!r}",
    )
