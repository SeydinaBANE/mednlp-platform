from src.core.exceptions import (
    AuthError,
    CollectionNotFoundError,
    DeidentificationError,
    DLQError,
    FineTuningError,
    GuardrailViolationError,
    InvalidTokenError,
    LLMError,
    ModelPromotionBlockedError,
    PipelineError,
    QualityGateError,
    VectorStoreError,
)


class TestDomainExceptions:
    def test_dlq_error(self) -> None:
        exc = DLQError("msg-1", "decode failed")
        assert exc.message_id == "msg-1"
        assert exc.reason == "decode failed"
        assert "msg-1" in str(exc)

    def test_pipeline_error(self) -> None:
        exc = PipelineError("segmenter", "OOM")
        assert exc.stage == "segmenter"
        assert exc.reason == "OOM"
        assert "segmenter" in str(exc)

    def test_quality_gate_error_is_pipeline_error(self) -> None:
        exc = QualityGateError("quality_gate", "expectation failed")
        assert isinstance(exc, PipelineError)

    def test_deidentification_error_is_pipeline_error(self) -> None:
        exc = DeidentificationError("deidentifier", "spacy failed")
        assert isinstance(exc, PipelineError)

    def test_collection_not_found_error(self) -> None:
        exc = CollectionNotFoundError("notes_v2")
        assert exc.collection == "notes_v2"
        assert "notes_v2" in str(exc)
        assert isinstance(exc, VectorStoreError)

    def test_guardrail_violation_error(self) -> None:
        exc = GuardrailViolationError("contains PII")
        assert exc.reason == "contains PII"
        assert "contains PII" in str(exc)

    def test_llm_error(self) -> None:
        exc = LLMError("anthropic/claude-3.5-sonnet", 429, "rate limited")
        assert exc.model == "anthropic/claude-3.5-sonnet"
        assert exc.status_code == 429
        assert exc.detail == "rate limited"
        assert "429" in str(exc)

    def test_model_promotion_blocked_error(self) -> None:
        exc = ModelPromotionBlockedError("icd10-v2", "f1", 0.72, 0.80)
        assert "icd10-v2" in str(exc)
        assert "0.7200" in str(exc)
        assert isinstance(exc, FineTuningError)

    def test_invalid_token_error_is_auth_error(self) -> None:
        exc = InvalidTokenError("token expired")
        assert isinstance(exc, AuthError)
        assert "token expired" in str(exc)
