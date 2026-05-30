"""Unit tests for embeddings: base, BiomedBERT, LoRA-Mistral, A/B router, registry."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.models import ABTest
from src.embeddings.ab_router import _bucket, dual_write, embed_and_index, route
from src.embeddings.base_embedder import BaseEmbedder
from src.embeddings.biomedbert_embedder import BiomedBertEmbedder
from src.embeddings.lora_mistral_embedder import LoraMistralEmbedder
from src.embeddings.registry import (
    get_model_by_version,
    get_production_model,
    list_registered_models,
)

# ── BaseEmbedder ──────────────────────────────────────────────────────────────


class _StubEmbedder(BaseEmbedder):
    """Minimal concrete embedder for testing the abstract base."""

    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def model_version(self) -> str:
        return "v0"

    @property
    def vector_size(self) -> int:
        return 4

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4]] * len(texts)


class TestBaseEmbedder:
    async def test_embed_one_delegates_to_embed(self) -> None:
        embedder = _StubEmbedder()
        vec = await embedder.embed_one("hello")
        assert vec == [0.1, 0.2, 0.3, 0.4]

    def test_abstract_properties(self) -> None:
        embedder = _StubEmbedder()
        assert embedder.model_name == "stub"
        assert embedder.vector_size == 4


# ── BiomedBertEmbedder ────────────────────────────────────────────────────────


class TestBiomedBertEmbedder:
    def test_properties(self) -> None:
        embedder = BiomedBertEmbedder()
        assert embedder.model_name == "biomedbert"
        assert embedder.vector_size == 768
        assert embedder.model_version == "v1"

    async def test_embed_calls_encode_sync(self) -> None:
        embedder = BiomedBertEmbedder()
        fake_vectors = [[0.1] * 768, [0.2] * 768]

        with patch.object(embedder, "_encode_sync", return_value=fake_vectors) as mock_enc:
            result = await embedder.embed(["note 1", "note 2"])

        mock_enc.assert_called_once_with(["note 1", "note 2"])
        assert result == fake_vectors

    def test_encode_sync_uses_cached_model(self) -> None:
        mock_model = MagicMock()
        import numpy as np

        mock_model.encode.return_value = np.array([[0.1] * 768])

        with patch("src.embeddings.biomedbert_embedder._load_model", return_value=mock_model):
            embedder = BiomedBertEmbedder()
            result = embedder._encode_sync(["clinical text"])

        mock_model.encode.assert_called_once()
        assert len(result) == 1
        assert len(result[0]) == 768


# ── LoraMistralEmbedder ───────────────────────────────────────────────────────


class TestLoraMistralEmbedder:
    def test_properties(self) -> None:
        embedder = LoraMistralEmbedder(
            mlflow_uri="http://localhost:5000",
            mlflow_model_name="lora-mistral-icd10",
            version="v2",
        )
        assert embedder.model_name == "lora-mistral"
        assert embedder.vector_size == 4096
        assert embedder.model_version == "v2"

    async def test_embed_calls_encode_sync(self) -> None:
        embedder = LoraMistralEmbedder(
            mlflow_uri="http://localhost:5000",
            mlflow_model_name="lora-mistral-icd10",
        )
        fake_vectors = [[0.1] * 4096]

        with patch.object(embedder, "_encode_sync", return_value=fake_vectors):
            result = await embedder.embed(["clinical note"])

        assert result == fake_vectors


# ── A/B Router ────────────────────────────────────────────────────────────────


def _make_ab_test(traffic_b_pct: float = 0.1) -> ABTest:
    return ABTest(
        name="test-ab",
        model_a="biomedbert",
        model_b="lora-mistral",
        traffic_b_pct=traffic_b_pct,
        is_active=True,
    )


class TestBucket:
    def test_is_deterministic(self) -> None:
        assert _bucket("note-001") == _bucket("note-001")

    def test_returns_0_to_99(self) -> None:
        for note_id in ["note-001", "note-002", "note-abc", "patient-xyz", "abc123"]:
            b = _bucket(note_id)
            assert 0 <= b <= 99

    def test_different_notes_get_different_buckets(self) -> None:
        buckets = {_bucket(f"note-{i}") for i in range(100)}
        assert len(buckets) > 50  # good distribution


class TestRoute:
    def test_routes_to_a_when_bucket_above_threshold(self) -> None:
        embedder_a = _StubEmbedder()
        embedder_b = _StubEmbedder()
        ab_test = _make_ab_test(traffic_b_pct=0.0)  # 0% to B — always A

        result = route("any-note", ab_test, embedder_a, embedder_b)
        assert result is embedder_a

    def test_routes_to_b_when_bucket_below_threshold(self) -> None:
        embedder_a = _StubEmbedder()
        embedder_b = _StubEmbedder()
        ab_test = _make_ab_test(traffic_b_pct=1.0)  # 100% to B — always B

        result = route("any-note", ab_test, embedder_a, embedder_b)
        assert result is embedder_b

    def test_same_note_always_routes_same_way(self) -> None:
        embedder_a = _StubEmbedder()
        embedder_b = _StubEmbedder()
        ab_test = _make_ab_test(traffic_b_pct=0.5)

        results = {route("stable-note-id", ab_test, embedder_a, embedder_b) for _ in range(10)}
        assert len(results) == 1  # always same model

    def test_traffic_split_approximately_correct(self) -> None:
        """With 10% B traffic, roughly 10% of 1000 notes should go to B."""
        embedder_a = _StubEmbedder()
        embedder_b = _StubEmbedder()
        ab_test = _make_ab_test(traffic_b_pct=0.1)

        count_b = sum(
            1
            for i in range(1000)
            if route(f"note-{i}", ab_test, embedder_a, embedder_b) is embedder_b
        )
        # Allow wide margin: 5–20%
        assert 50 <= count_b <= 200


class TestEmbedAndIndex:
    async def test_partitions_and_indexes(self) -> None:
        embedder_a = _StubEmbedder()
        embedder_b = _StubEmbedder()
        ab_test = _make_ab_test(traffic_b_pct=0.0)  # All to A

        with patch("src.embeddings.ab_router.upsert_notes", new_callable=AsyncMock) as mock_upsert:
            mock_upsert.return_value = 2
            counts = await embed_and_index(
                note_ids=["n1", "n2"],
                texts=["text 1", "text 2"],
                payloads=[{}, {}],
                ab_test=ab_test,
                embedder_a=embedder_a,
                embedder_b=embedder_b,
            )

        assert counts.get("stub", 0) == 2

    async def test_dual_write_indexes_both_models(self) -> None:
        embedder_a = _StubEmbedder()
        embedder_b = _StubEmbedder()

        with patch("src.embeddings.ab_router.upsert_notes", new_callable=AsyncMock) as mock_upsert:
            mock_upsert.return_value = 2
            await dual_write(
                note_ids=["n1", "n2"],
                texts=["text 1", "text 2"],
                payloads=[{}, {}],
                embedder_a=embedder_a,
                embedder_b=embedder_b,
            )

        assert mock_upsert.call_count == 2


# ── Registry ──────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_get_production_model(self) -> None:
        mock_mv = MagicMock()
        mock_mv.name = "lora-mistral-icd10"
        mock_mv.version = "3"
        mock_mv.current_stage = "Production"
        mock_mv.run_id = "run-abc"
        mock_mv.source = "s3://bucket/model"

        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = [mock_mv]

        with patch("src.embeddings.registry._get_mlflow_client", return_value=mock_client):
            info = get_production_model("lora-mistral-icd10")

        assert info.name == "lora-mistral-icd10"
        assert info.version == "3"
        assert info.stage == "Production"

    def test_get_production_model_raises_when_none(self) -> None:
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []

        with patch("src.embeddings.registry._get_mlflow_client", return_value=mock_client):
            with pytest.raises(ValueError, match="No Production version"):
                get_production_model("missing-model")

    def test_list_registered_models(self) -> None:
        m1, m2 = MagicMock(), MagicMock()
        m1.name = "biomedbert-base"
        m2.name = "lora-mistral-icd10"

        mock_client = MagicMock()
        mock_client.search_registered_models.return_value = [m1, m2]

        with patch("src.embeddings.registry._get_mlflow_client", return_value=mock_client):
            names = list_registered_models()

        assert "biomedbert-base" in names
        assert "lora-mistral-icd10" in names

    def test_get_model_by_version(self) -> None:
        mock_mv = MagicMock()
        mock_mv.name = "lora-mistral-icd10"
        mock_mv.version = "2"
        mock_mv.current_stage = "Staging"
        mock_mv.run_id = "run-xyz"
        mock_mv.source = "s3://bucket/model/v2"

        mock_client = MagicMock()
        mock_client.get_model_version.return_value = mock_mv

        with patch("src.embeddings.registry._get_mlflow_client", return_value=mock_client):
            info = get_model_by_version("lora-mistral-icd10", "2")

        assert info.version == "2"
        assert info.stage == "Staging"
