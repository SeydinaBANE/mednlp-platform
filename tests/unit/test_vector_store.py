"""Unit tests for vector_store: collections, indexer, search."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import CollectionNotFoundError, VectorStoreError
from src.vector_store.client import get_qdrant_client
from src.vector_store.collections import (
    _resolve_vector_size,
    collection_name,
    ensure_collection,
    get_collection_info,
    list_note_collections,
)
from src.vector_store.indexer import _make_batches, _stable_point_id, upsert_notes
from src.vector_store.search import _build_filter, search_multi_collection, search_similar

# ── collections ───────────────────────────────────────────────────────────────


class TestCollectionName:
    def test_basic(self) -> None:
        assert collection_name("biomedbert", "v1") == "notes_biomedbert_v1"

    def test_slashes_replaced(self) -> None:
        assert collection_name("microsoft/BiomedBERT", "v2") == "notes_microsoft_biomedbert_v2"

    def test_hyphens_replaced(self) -> None:
        assert collection_name("lora-mistral", "v1") == "notes_lora_mistral_v1"

    def test_version_dots_replaced(self) -> None:
        assert collection_name("biomedbert", "1.2.3") == "notes_biomedbert_1_2_3"


class TestResolveVectorSize:
    def test_biomedbert(self) -> None:
        assert _resolve_vector_size("biomedbert") == 768

    def test_lora_mistral(self) -> None:
        assert _resolve_vector_size("lora-mistral") == 4096

    def test_unknown_raises(self) -> None:
        with pytest.raises(VectorStoreError):
            _resolve_vector_size("unknown-model-xyz")


class TestEnsureCollection:
    async def test_returns_existing_collection(self) -> None:
        mock_client = AsyncMock()
        mock_client.collection_exists.return_value = True

        with patch("src.vector_store.collections.get_qdrant_client", return_value=mock_client):
            name = await ensure_collection("biomedbert", "v1")

        assert name == "notes_biomedbert_v1"
        mock_client.create_collection.assert_not_called()

    async def test_creates_new_collection(self) -> None:
        mock_client = AsyncMock()
        mock_client.collection_exists.return_value = False

        with patch("src.vector_store.collections.get_qdrant_client", return_value=mock_client):
            name = await ensure_collection("biomedbert", "v1")

        assert name == "notes_biomedbert_v1"
        mock_client.create_collection.assert_awaited_once()

    async def test_raises_vector_store_error_on_failure(self) -> None:
        mock_client = AsyncMock()
        mock_client.collection_exists.side_effect = RuntimeError("qdrant down")

        with patch("src.vector_store.collections.get_qdrant_client", return_value=mock_client):
            with pytest.raises(VectorStoreError):
                await ensure_collection("biomedbert", "v1")


class TestGetCollectionInfo:
    async def test_returns_info(self) -> None:
        mock_info = MagicMock()
        mock_info.points_count = 1000
        mock_info.indexed_vectors_count = 950
        mock_info.status = "green"

        mock_client = AsyncMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = mock_info

        with patch("src.vector_store.collections.get_qdrant_client", return_value=mock_client):
            info = await get_collection_info("notes_biomedbert_v1")

        assert info["points_count"] == 1000
        assert info["name"] == "notes_biomedbert_v1"

    async def test_raises_not_found_when_missing(self) -> None:
        mock_client = AsyncMock()
        mock_client.collection_exists.return_value = False

        with patch("src.vector_store.collections.get_qdrant_client", return_value=mock_client):
            with pytest.raises(CollectionNotFoundError):
                await get_collection_info("notes_biomedbert_v1")


class TestListNoteCollections:
    async def test_filters_to_notes_prefix(self) -> None:
        c1, c2, c3 = MagicMock(), MagicMock(), MagicMock()
        c1.name = "notes_biomedbert_v1"
        c2.name = "notes_lora_mistral_v1"
        c3.name = "other_collection"

        mock_result = MagicMock()
        mock_result.collections = [c1, c2, c3]

        mock_client = AsyncMock()
        mock_client.get_collections.return_value = mock_result

        with patch("src.vector_store.collections.get_qdrant_client", return_value=mock_client):
            names = await list_note_collections()

        assert "notes_biomedbert_v1" in names
        assert "notes_lora_mistral_v1" in names
        assert "other_collection" not in names


# ── indexer ───────────────────────────────────────────────────────────────────


class TestStablePointId:
    def test_is_deterministic(self) -> None:
        assert _stable_point_id("note-001") == _stable_point_id("note-001")

    def test_different_notes_get_different_ids(self) -> None:
        assert _stable_point_id("note-001") != _stable_point_id("note-002")

    def test_returns_valid_uuid_string(self) -> None:
        import uuid

        pid = _stable_point_id("note-abc")
        uuid.UUID(pid)  # raises if not valid UUID


class TestMakeBatches:
    def test_single_batch(self) -> None:
        ids = ["n1", "n2", "n3"]
        vecs = [[0.1], [0.2], [0.3]]
        payloads = [{}, {}, {}]
        batches = _make_batches(ids, vecs, payloads, batch_size=10)
        assert len(batches) == 1
        assert batches[0][0] == ids

    def test_multiple_batches(self) -> None:
        ids = [f"n{i}" for i in range(5)]
        vecs = [[float(i)] for i in range(5)]
        payloads = [{} for _ in range(5)]
        batches = _make_batches(ids, vecs, payloads, batch_size=2)
        assert len(batches) == 3
        assert len(batches[0][0]) == 2
        assert len(batches[2][0]) == 1  # last batch has 1 item


class TestUpsertNotes:
    async def test_upserts_successfully(self) -> None:
        mock_client = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.return_value = "notes_biomedbert_v1"

        with (
            patch("src.vector_store.indexer.ensure_collection", return_value="notes_biomedbert_v1"),
            patch("src.vector_store.indexer.get_qdrant_client", return_value=mock_client),
        ):
            total = await upsert_notes(
                note_ids=["n1", "n2"],
                vectors=[[0.1] * 768, [0.2] * 768],
                payloads=[{"patient_id": "p1"}, {"patient_id": "p2"}],
                model_name="biomedbert",
                model_version="v1",
            )

        assert total == 2
        mock_client.upsert.assert_awaited_once()

    async def test_raises_on_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            await upsert_notes(
                note_ids=["n1"],
                vectors=[[0.1] * 768, [0.2] * 768],
                payloads=[{}],
                model_name="biomedbert",
                model_version="v1",
            )

    async def test_raises_vector_store_error_on_upsert_failure(self) -> None:
        mock_client = AsyncMock()
        mock_client.upsert.side_effect = RuntimeError("connection refused")

        with (
            patch("src.vector_store.indexer.ensure_collection", return_value="notes_biomedbert_v1"),
            patch("src.vector_store.indexer.get_qdrant_client", return_value=mock_client),
        ):
            with pytest.raises(VectorStoreError):
                await upsert_notes(
                    note_ids=["n1"],
                    vectors=[[0.1] * 768],
                    payloads=[{}],
                    model_name="biomedbert",
                    model_version="v1",
                )


# ── search ────────────────────────────────────────────────────────────────────


class TestBuildFilter:
    def test_no_filters(self) -> None:
        assert _build_filter(None, None) is None

    def test_patient_id_filter(self) -> None:
        f = _build_filter("patient-1", None)
        assert f is not None
        assert len(f.must) == 1

    def test_both_filters(self) -> None:
        f = _build_filter("patient-1", "progress_note")
        assert f is not None
        assert len(f.must) == 2


class TestSearchSimilar:
    async def test_returns_results(self) -> None:
        hit = MagicMock()
        hit.id = "point-1"
        hit.score = 0.92
        hit.payload = {"note_id": "note-001", "patient_id": "p1"}

        mock_client = AsyncMock()
        mock_client.search.return_value = [hit]

        with patch("src.vector_store.search.get_qdrant_client", return_value=mock_client):
            results = await search_similar(
                query_vector=[0.1] * 768,
                collection="notes_biomedbert_v1",
                top_k=5,
            )

        assert len(results) == 1
        assert results[0].note_id == "note-001"
        assert results[0].score == 0.92

    async def test_raises_collection_not_found(self) -> None:
        mock_client = AsyncMock()
        mock_client.search.side_effect = RuntimeError("collection not found")

        with patch("src.vector_store.search.get_qdrant_client", return_value=mock_client):
            with pytest.raises(CollectionNotFoundError):
                await search_similar(
                    query_vector=[0.1] * 768,
                    collection="notes_missing",
                )

    async def test_raises_vector_store_error_on_other_failure(self) -> None:
        mock_client = AsyncMock()
        mock_client.search.side_effect = RuntimeError("timeout")

        with patch("src.vector_store.search.get_qdrant_client", return_value=mock_client):
            with pytest.raises(VectorStoreError):
                await search_similar(
                    query_vector=[0.1] * 768,
                    collection="notes_biomedbert_v1",
                )

    async def test_filters_by_patient(self) -> None:
        mock_client = AsyncMock()
        mock_client.search.return_value = []

        with patch("src.vector_store.search.get_qdrant_client", return_value=mock_client):
            await search_similar(
                query_vector=[0.1] * 768,
                collection="notes_biomedbert_v1",
                patient_id="patient-42",
            )

        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs["query_filter"] is not None


class TestGetQdrantClient:
    def test_returns_client_instance(self) -> None:
        get_qdrant_client.cache_clear()
        with patch("src.vector_store.client.AsyncQdrantClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            client = get_qdrant_client()
            assert client is not None
        get_qdrant_client.cache_clear()

    def test_is_cached(self) -> None:
        get_qdrant_client.cache_clear()
        with patch("src.vector_store.client.AsyncQdrantClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            c1 = get_qdrant_client()
            c2 = get_qdrant_client()
            assert c1 is c2
        get_qdrant_client.cache_clear()


class TestSearchMultiCollection:
    async def test_merges_results_from_multiple_collections(self) -> None:
        hit_a = MagicMock()
        hit_a.id = "pa"
        hit_a.score = 0.95
        hit_a.payload = {"note_id": "note-a"}

        hit_b = MagicMock()
        hit_b.id = "pb"
        hit_b.score = 0.80
        hit_b.payload = {"note_id": "note-b"}

        mock_client = AsyncMock()
        mock_client.search.side_effect = [[hit_a], [hit_b]]

        with patch("src.vector_store.search.get_qdrant_client", return_value=mock_client):
            results = await search_multi_collection(
                [0.1] * 768,
                ["notes_biomedbert_v1", "notes_lora_mistral_v1"],
                top_k=5,
            )

        assert len(results) == 2
        assert results[0].score >= results[1].score  # sorted by score desc
