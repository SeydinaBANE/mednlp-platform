"""Shared dataclasses for pipeline stage inputs/outputs."""

from dataclasses import dataclass, field
from typing import Any

from src.ingestion.schemas import NoteRecord


@dataclass
class Segment:
    text: str
    start_char: int
    end_char: int
    sentence_index: int


@dataclass
class Entity:
    text: str
    label: str  # e.g. DISEASE, CHEMICAL, PROCEDURE
    start_char: int
    end_char: int
    score: float
    umls_cui: str | None = None


@dataclass
class QualityGateResult:
    passed: bool
    suite_name: str
    failed_expectations: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """Carries accumulated state as a note moves through the pipeline."""

    note: NoteRecord
    segments: list[Segment] = field(default_factory=list)
    deidentified_text: str = ""
    entities: list[Entity] = field(default_factory=list)
    quality: QualityGateResult | None = None
    vector_indexed: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def processed_text(self) -> str:
        return self.deidentified_text or self.note.raw_text
