"""Unit tests for fine_tuning: data_prep, evaluator, lora_config, promote_model, vertex_job."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from mlflow.exceptions import MlflowException

from src.core.exceptions import ModelPromotionBlockedError
from src.fine_tuning.data_prep import (
    _jaccard_estimate,
    _minhash_signature,
    _shingles,
    build_icd10_dataset,
    build_triage_dataset,
    minhash_dedup,
    stratified_split,
)
from src.fine_tuning.evaluator import bootstrap_ci, evaluate_icd10, evaluate_triage
from src.fine_tuning.lora_config import LoraHyperparams, get_icd10_config, get_peft_config
from src.fine_tuning.promote_model import promote_icd10, promote_triage
from src.fine_tuning.vertex_job import VertexJobConfig, submit_fine_tune_job

# ── data_prep ─────────────────────────────────────────────────────────────────


class TestShingles:
    def test_basic_shingles(self) -> None:
        s = _shingles("the patient presents with pain", k=3)
        assert "the patient presents" in s
        assert "patient presents with" in s

    def test_short_text_returns_full_text(self) -> None:
        s = _shingles("short", k=3)
        assert s == {"short"}

    def test_case_insensitive(self) -> None:
        s = _shingles("Hello WORLD foo", k=2)
        assert "hello world" in s


class TestMinHashSignature:
    def test_returns_correct_length(self) -> None:
        sig = _minhash_signature("test text", num_perm=64)
        assert len(sig) == 64

    def test_is_deterministic(self) -> None:
        text = "patient presents with acute chest pain"
        assert _minhash_signature(text) == _minhash_signature(text)

    def test_similar_texts_have_similar_sigs(self) -> None:
        sig1 = _minhash_signature("patient has chest pain and shortness of breath")
        sig2 = _minhash_signature("patient has chest pain and shortness of breath")
        est = _jaccard_estimate(sig1, sig2)
        assert est == 1.0

    def test_different_texts_have_lower_similarity(self) -> None:
        sig1 = _minhash_signature("patient has fever and cough")
        sig2 = _minhash_signature("orthopedic surgery for fracture repair")
        est = _jaccard_estimate(sig1, sig2)
        assert est < 0.9


class TestMinHashDedup:
    def test_keeps_all_unique_texts(self) -> None:
        texts = ["text one here", "completely different text", "another unique note"]
        mask = minhash_dedup(texts, threshold=0.9)
        assert all(mask)

    def test_removes_exact_duplicates(self) -> None:
        texts = ["exact same text"] * 5
        mask = minhash_dedup(texts, threshold=0.8)
        assert mask.count(True) == 1

    def test_keeps_moderately_different_texts(self) -> None:
        texts = [
            "patient has fever and cough",
            "patient admitted with fracture of left femur",
        ]
        mask = minhash_dedup(texts, threshold=0.8)
        assert mask.count(True) == 2


class TestStratifiedSplit:
    def _make_records(self, n: int, labels: list[str]) -> list[dict[str, object]]:
        return [{"text": f"note {i}", "label": labels[i % len(labels)]} for i in range(n)]

    def test_correct_split_sizes(self) -> None:
        records = self._make_records(100, ["A", "B"])
        train, val, test = stratified_split(records, "label", ratios=(0.7, 0.15, 0.15))
        assert len(train) + len(val) + len(test) == 100
        assert len(train) > len(val)
        assert len(val) > 0
        assert len(test) > 0

    def test_invalid_ratios_raise(self) -> None:
        with pytest.raises(AssertionError):
            stratified_split([], "label", ratios=(0.5, 0.5, 0.5))

    def test_deterministic_with_seed(self) -> None:
        records = self._make_records(50, ["A", "B", "C"])
        split1 = stratified_split(records, "label", seed=42)
        split2 = stratified_split(records, "label", seed=42)
        assert [r["text"] for r in split1[0]] == [r["text"] for r in split2[0]]


class TestDatasetBuilders:
    def test_build_icd10_dataset(self) -> None:
        records = [
            {"raw_text": "note text", "icd10_codes": ["I10", "E11"]},
        ]
        label2id = {"I10": 0, "E11": 1, "J18": 2}
        ds = build_icd10_dataset(records, label2id)
        assert len(ds) == 1
        assert ds[0]["labels"][0] == 1.0
        assert ds[0]["labels"][1] == 1.0
        assert ds[0]["labels"][2] == 0.0

    def test_build_triage_dataset(self) -> None:
        records = [
            {"raw_text": "critical patient", "esi_level": 1},
            {"raw_text": "routine visit", "esi_level": 5},
        ]
        ds = build_triage_dataset(records)
        assert len(ds) == 2
        assert ds[0]["labels"] == 0  # ESI 1 → class 0
        assert ds[1]["labels"] == 4  # ESI 5 → class 4

    def test_build_triage_clamps_esi(self) -> None:
        records = [{"raw_text": "note", "esi_level": 10}]
        ds = build_triage_dataset(records)
        assert ds[0]["labels"] == 4  # clamped to max


# ── evaluator ─────────────────────────────────────────────────────────────────


class TestBootstrapCI:
    def test_returns_tuple_of_two_floats(self) -> None:
        scores = [0.8, 0.85, 0.82, 0.79, 0.84]
        lo, hi = bootstrap_ci(scores)
        assert lo <= hi
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_is_deterministic_with_seed(self) -> None:
        scores = [0.7, 0.75, 0.72]
        r1 = bootstrap_ci(scores, seed=42)
        r2 = bootstrap_ci(scores, seed=42)
        assert r1 == r2


class TestEvaluateICD10:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_prob = np.array([[0.9, 0.1, 0.8], [0.1, 0.9, 0.1]])
        result = evaluate_icd10(y_true, y_prob)
        assert result["f1_macro"] == pytest.approx(1.0)

    def test_zero_predictions(self) -> None:
        y_true = np.array([[1, 0], [0, 1]])
        y_prob = np.array([[0.1, 0.1], [0.1, 0.1]])
        result = evaluate_icd10(y_true, y_prob)
        assert result["f1_macro"] == pytest.approx(0.0)

    def test_label_names_in_result(self) -> None:
        y_true = np.array([[1, 0]])
        y_prob = np.array([[0.9, 0.1]])
        result = evaluate_icd10(y_true, y_prob, label_names=["I10", "E11"])
        assert "per_label_f1" in result
        assert "I10" in result["per_label_f1"]


class TestEvaluateTriage:
    def test_perfect_predictions(self) -> None:
        y_true = [0, 1, 2, 3, 4]
        y_pred = [0, 1, 2, 3, 4]
        result = evaluate_triage(y_true, y_pred)
        assert result["f1_weighted"] == pytest.approx(1.0)
        assert result["f1_macro"] == pytest.approx(1.0)

    def test_class_names_in_result(self) -> None:
        result = evaluate_triage([0, 1], [0, 1], class_names=["ESI1", "ESI2"])
        assert "per_class_f1" in result


# ── lora_config ───────────────────────────────────────────────────────────────


class TestLoraConfig:
    def test_default_hyperparams(self) -> None:
        hp = LoraHyperparams()
        assert hp.r == 16
        assert hp.lora_alpha == 32
        assert hp.bias == "none"

    def test_get_peft_config_icd10(self) -> None:
        from peft import LoraConfig

        config = get_icd10_config()
        assert isinstance(config, LoraConfig)
        assert config.r == 16

    def test_get_peft_config_custom_hp(self) -> None:
        from peft import LoraConfig

        hp = LoraHyperparams(r=8, lora_alpha=16)
        config = get_peft_config("triage", hp)
        assert isinstance(config, LoraConfig)
        assert config.r == 8

    def test_invalid_task_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown task"):
            get_peft_config("unknown_task")


# ── promote_model ─────────────────────────────────────────────────────────────


class TestPromoteModel:
    def _mock_mlflow_client(self, version: str = "3") -> MagicMock:
        mock_mv = MagicMock()
        mock_mv.version = version
        mock_client = MagicMock()
        mock_client.get_model_version_by_alias.return_value = mock_mv
        return mock_client

    def test_promotes_icd10_when_f1_passes(self) -> None:
        y_true = np.eye(3)
        y_prob = np.eye(3) * 0.9 + 0.05
        mock_client = self._mock_mlflow_client()

        with patch("src.fine_tuning.promote_model._get_mlflow_client", return_value=mock_client):
            result = promote_icd10("icd10-model", y_true, y_prob, threshold=0.5)

        assert result["promoted"] is True
        mock_client.set_registered_model_alias.assert_called_once()

    def test_blocks_icd10_when_f1_fails(self) -> None:
        y_true = np.eye(3)
        y_prob = np.zeros((3, 3))  # all zeros → F1=0
        mock_client = self._mock_mlflow_client()

        with patch("src.fine_tuning.promote_model._get_mlflow_client", return_value=mock_client):
            with pytest.raises(ModelPromotionBlockedError):
                promote_icd10("icd10-model", y_true, y_prob, threshold=0.8)

    def test_promotes_triage_when_f1_passes(self) -> None:
        y_true = [0, 1, 2, 3, 4]
        y_pred = [0, 1, 2, 3, 4]
        mock_client = self._mock_mlflow_client()

        with patch("src.fine_tuning.promote_model._get_mlflow_client", return_value=mock_client):
            result = promote_triage("triage-model", y_true, y_pred, threshold=0.5)

        assert result["promoted"] is True

    def test_raises_when_no_staging_model(self) -> None:
        mock_client = MagicMock()
        mock_client.get_model_version_by_alias.side_effect = MlflowException("alias not found")

        with patch("src.fine_tuning.promote_model._get_mlflow_client", return_value=mock_client):
            with pytest.raises(ValueError, match="No staging alias"):
                promote_icd10("missing-model", np.zeros((1, 1)), np.zeros((1, 1)))


# ── vertex_job ────────────────────────────────────────────────────────────────


def _patch_vertex(
    mock_job_resource: str = "projects/p/trainingPipelines/123",
) -> tuple[MagicMock, dict[str, MagicMock]]:
    mock_aiplatform = MagicMock()
    mock_job = MagicMock()
    mock_job.resource_name = mock_job_resource
    mock_aiplatform.CustomContainerTrainingJob.return_value = mock_job
    modules = {"google.cloud.aiplatform": mock_aiplatform}
    return mock_aiplatform, modules


class TestVertexJob:
    def test_invalid_task_raises(self) -> None:
        config = VertexJobConfig(task="invalid_task")
        _, modules = _patch_vertex()

        with (
            patch.dict("sys.modules", modules),
            patch("src.fine_tuning.vertex_job.get_settings") as mock_settings,
        ):
            mock_settings.return_value.gcp_project_id = "test-project"
            mock_settings.return_value.gcp_region = "us-central1"
            mock_settings.return_value.gcs_bucket_artifacts = "bucket"
            with pytest.raises(ValueError, match="task must be one of"):
                submit_fine_tune_job(config)

    def test_valid_task_submits(self) -> None:
        config = VertexJobConfig(task="icd10", sync=False)
        _, modules = _patch_vertex()

        with (
            patch.dict("sys.modules", modules),
            patch("src.fine_tuning.vertex_job.get_settings") as mock_settings,
        ):
            mock_settings.return_value.gcp_project_id = "test-project"
            mock_settings.return_value.gcp_region = "us-central1"
            mock_settings.return_value.gcs_bucket_artifacts = "mednlp-artifacts"
            job_name = submit_fine_tune_job(config)

        assert job_name == "projects/p/trainingPipelines/123"

    def test_triage_task_submits(self) -> None:
        config = VertexJobConfig(task="triage", display_name="my-triage-job")
        _, modules = _patch_vertex()

        with (
            patch.dict("sys.modules", modules),
            patch("src.fine_tuning.vertex_job.get_settings") as mock_settings,
        ):
            mock_settings.return_value.gcp_project_id = "test-project"
            mock_settings.return_value.gcp_region = "us-central1"
            mock_settings.return_value.gcs_bucket_artifacts = "bucket"
            job_name = submit_fine_tune_job(config)

        assert job_name == "projects/p/trainingPipelines/123"

    def test_main_exits_on_invalid_task(self) -> None:
        import os

        from src.fine_tuning.vertex_job import main

        with patch.dict(os.environ, {"TASK": "invalid"}):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
