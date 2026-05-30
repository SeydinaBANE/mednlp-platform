"""Pydantic v2 request/response schemas for the MedNLP API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.ingestion.schemas import NoteType

# ── Shared ────────────────────────────────────────────────────────────────────


class PaginatedResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[Any]


# ── Note ──────────────────────────────────────────────────────────────────────


class NoteResponse(BaseModel):
    id: str
    patient_id: str
    encounter_id: str | None
    note_type: NoteType
    authored_at: datetime
    source: str
    is_deidentified: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class NoteDetailResponse(NoteResponse):
    raw_text: str
    processed_text: str | None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"from_attributes": True, "populate_by_name": True}


# ── Pipeline ──────────────────────────────────────────────────────────────────


class PipelineJobResponse(BaseModel):
    id: int
    note_id: str
    status: str
    stage: str | None
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None
    duration_ms: int | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── RAG Query ─────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str = Field(min_length=5, max_length=2000)
    patient_id: str | None = Field(default=None, description="Limit retrieval to a single patient")
    top_k: int = Field(default=5, ge=1, le=20)
    model: str | None = Field(default=None, description="Override LLM model slug")
    stream: bool = Field(default=False)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class CitationSource(BaseModel):
    note_id: str
    patient_id: str
    note_type: NoteType
    authored_at: datetime
    excerpt: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[CitationSource]
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int


# ── Ingestion ─────────────────────────────────────────────────────────────────


class IngestResponse(BaseModel):
    note_id: str
    job_id: int
    status: str


# ── Auth ──────────────────────────────────────────────────────────────────────


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# ── Audit ─────────────────────────────────────────────────────────────────────


class AuditLogResponse(BaseModel):
    id: int
    actor: str
    action: str
    resource_type: str
    resource_id: str | None
    ip_address: str | None
    details: dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Health ────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    version: str
    db: str
    redis: str
    qdrant: str
