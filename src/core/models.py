"""SQLAlchemy 2.0 ORM — declarative models for MedNLP Platform."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from src.ingestion.schemas import NoteType


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class Base(DeclarativeBase):
    type_annotation_map: dict[type, Any] = {}


# ── NoteRecord ────────────────────────────────────────────────────────────────


class Note(Base):
    """Persisted clinical note after ingestion."""

    __tablename__ = "notes"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    patient_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    encounter_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    note_type: Mapped[str] = mapped_column(
        Enum(NoteType, name="note_type_enum", values_callable=lambda e: [v.value for v in e]),
        nullable=False,
        default=NoteType.UNKNOWN.value,
    )
    authored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    processed_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, default="fhir")
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    is_deidentified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    jobs: Mapped[list["PipelineJob"]] = relationship(back_populates="note", lazy="select")
    embeddings: Mapped[list["EmbeddingRecord"]] = relationship(back_populates="note", lazy="select")

    __table_args__ = (
        Index("ix_notes_patient_authored", "patient_id", "authored_at"),
        Index("ix_notes_source_type", "source", "note_type"),
    )


# ── PipelineJob ───────────────────────────────────────────────────────────────


class PipelineJob(Base):
    """Tracks a single pipeline execution for a note."""

    __tablename__ = "pipeline_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    note_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("notes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending"
    )  # pending | running | success | failed
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    note: Mapped["Note"] = relationship(back_populates="jobs")

    __table_args__ = (Index("ix_jobs_status_created", "status", "created_at"),)


# ── EmbeddingRecord ───────────────────────────────────────────────────────────


class EmbeddingRecord(Base):
    """Tracks which vector was indexed for a note and with which model version."""

    __tablename__ = "embedding_records"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    note_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("notes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    qdrant_collection: Mapped[str] = mapped_column(String(128), nullable=False)
    qdrant_point_id: Mapped[str] = mapped_column(String(256), nullable=False)
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    note: Mapped["Note"] = relationship(back_populates="embeddings")

    __table_args__ = (
        Index("ix_embeddings_model_version", "model_name", "model_version"),
        Index("ix_embeddings_collection", "qdrant_collection"),
    )


# ── ABTest ────────────────────────────────────────────────────────────────────


class ABTest(Base):
    """Active A/B experiments controlling embedding model routing."""

    __tablename__ = "ab_tests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    model_a: Mapped[str] = mapped_column(String(128), nullable=False)
    model_b: Mapped[str] = mapped_column(String(128), nullable=False)
    traffic_b_pct: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.1
    )  # fraction of traffic to model_b (0.0–1.0)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# ── DriftEvent ────────────────────────────────────────────────────────────────


class DriftEvent(Base):
    """Records a detected drift alert from Evidently or KS tests."""

    __tablename__ = "drift_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    drift_type: Mapped[str] = mapped_column(String(32), nullable=False)  # embedding | label | data
    metric_name: Mapped[str] = mapped_column(String(128), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    alerted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )


# ── AuditLog ──────────────────────────────────────────────────────────────────


class AuditLog(Base):
    """HIPAA-compliant audit trail for all note accesses and mutations."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(256), nullable=False)  # user_id or service name
    action: Mapped[str] = mapped_column(String(64), nullable=False)  # read | write | delete | query
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)  # note | query | job
    resource_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )

    __table_args__ = (
        Index("ix_audit_actor_created", "actor", "created_at"),
        Index("ix_audit_resource", "resource_type", "resource_id"),
    )
