from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class NoteType(StrEnum):
    PROGRESS_NOTE = "progress_note"
    DISCHARGE_SUMMARY = "discharge_summary"
    RADIOLOGY_REPORT = "radiology_report"
    OPERATIVE_REPORT = "operative_report"
    CONSULTATION = "consultation"
    UNKNOWN = "unknown"


class NoteRecord(BaseModel):
    """Internal representation of a clinical note, source-agnostic."""

    note_id: str = Field(description="Unique identifier (UUID)")
    patient_id: str = Field(description="Patient reference (pseudonymised in dev)")
    encounter_id: str | None = Field(default=None)
    note_type: NoteType = Field(default=NoteType.UNKNOWN)
    authored_at: datetime
    raw_text: str = Field(description="Original note text before any processing")
    source: str = Field(default="fhir", description="ingestion source: fhir | batch | api")
    metadata: dict[str, str] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """Direct REST ingestion payload."""

    patient_id: str
    note_type: NoteType = NoteType.UNKNOWN
    text: str = Field(min_length=10)
    authored_at: datetime | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
