class MedNLPError(Exception):
    """Base exception for all domain errors."""


# ── Ingestion ─────────────────────────────────────────────────────────────────


class IngestionError(MedNLPError):
    """Raised when note ingestion fails."""


class FHIRParseError(IngestionError):
    """Raised when a FHIR resource cannot be parsed."""

    def __init__(self, resource_id: str, reason: str) -> None:
        self.resource_id = resource_id
        self.reason = reason
        super().__init__(f"Cannot parse FHIR resource {resource_id!r}: {reason}")


class MissingPatientReferenceError(IngestionError):
    """Raised when a FHIR resource has no subject (patient) reference."""

    def __init__(self, resource_id: str) -> None:
        self.resource_id = resource_id
        super().__init__(f"FHIR resource {resource_id!r} has no subject reference")


class DLQError(IngestionError):
    """Raised when a message must be sent to the dead-letter queue."""

    def __init__(self, message_id: str, reason: str) -> None:
        self.message_id = message_id
        self.reason = reason
        super().__init__(f"Message {message_id!r} sent to DLQ: {reason}")


# ── Pipeline ──────────────────────────────────────────────────────────────────


class PipelineError(MedNLPError):
    """Raised when a processing stage fails."""

    def __init__(self, stage: str, reason: str) -> None:
        self.stage = stage
        self.reason = reason
        super().__init__(f"Stage {stage!r} failed: {reason}")


class QualityGateError(PipelineError):
    """Raised when a Great Expectations quality check fails."""


class DeidentificationError(PipelineError):
    """Raised when PHI de-identification cannot be completed."""


# ── Vector store ──────────────────────────────────────────────────────────────


class VectorStoreError(MedNLPError):
    """Raised when Qdrant operations fail."""


class CollectionNotFoundError(VectorStoreError):
    """Raised when a required Qdrant collection does not exist."""

    def __init__(self, collection: str) -> None:
        self.collection = collection
        super().__init__(f"Qdrant collection {collection!r} not found")


# ── RAG ───────────────────────────────────────────────────────────────────────


class RAGError(MedNLPError):
    """Raised when RAG retrieval or generation fails."""


class GuardrailViolationError(RAGError):
    """Raised when the answer generator output violates a safety guardrail."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Guardrail violation: {reason}")


# ── LLM ───────────────────────────────────────────────────────────────────────


class LLMError(MedNLPError):
    """Raised when an OpenRouter call fails."""

    def __init__(self, model: str, status_code: int, detail: str) -> None:
        self.model = model
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"LLM error [{status_code}] on {model!r}: {detail}")


# ── Fine-tuning ───────────────────────────────────────────────────────────────


class FineTuningError(MedNLPError):
    """Raised when fine-tuning pipeline operations fail."""


class ModelPromotionBlockedError(FineTuningError):
    """Raised when a model fails the evaluation gate."""

    def __init__(self, model_name: str, metric: str, actual: float, threshold: float) -> None:
        super().__init__(
            f"Model {model_name!r} blocked from promotion: "
            f"{metric}={actual:.4f} < threshold {threshold:.4f}"
        )


# ── Auth ──────────────────────────────────────────────────────────────────────


class AuthError(MedNLPError):
    """Raised on authentication/authorization failures."""


class InvalidTokenError(AuthError):
    """Raised when a JWT token is invalid or expired."""
