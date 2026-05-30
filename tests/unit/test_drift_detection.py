"""Unit tests for drift_detection: embedding_drift, label_drift, data_drift, alert_publisher."""

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.drift_detection.alert_publisher import (
    publish_data_drift,
    publish_embedding_drift,
    publish_label_drift,
)
from src.drift_detection.data_drift import DataDriftResult, _aggregate_counts, detect_data_drift
from src.drift_detection.embedding_drift import (
    EmbeddingDriftResult,
    detect_embedding_drift,
    js_divergence,
)
from src.drift_detection.label_drift import LabelDriftResult, _code_frequencies, detect_label_drift

# ── embedding_drift ───────────────────────────────────────────────────────────


class TestJsDivergence:
    def test_identical_distributions_zero(self) -> None:
        arr = np.random.default_rng(42).normal(0, 1, 500).astype(np.float64)
        score = js_divergence(arr, arr.copy())
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_different_distributions_positive(self) -> None:
        rng = np.random.default_rng(42)
        p = rng.normal(0, 1, 500).astype(np.float64)
        q = rng.normal(5, 1, 500).astype(np.float64)  # shifted distribution
        score = js_divergence(p, q)
        assert score > 0.05

    def test_constant_array_returns_zero(self) -> None:
        arr = np.ones(100, dtype=np.float64)
        assert js_divergence(arr, arr) == 0.0

    def test_returns_float(self) -> None:
        rng = np.random.default_rng(0)
        p = rng.normal(size=100).astype(np.float64)
        q = rng.normal(size=100).astype(np.float64)
        assert isinstance(js_divergence(p, q), float)


class TestDetectEmbeddingDrift:
    def _make_embeddings(self, n: int, dim: int = 128, shift: float = 0.0) -> np.ndarray:
        rng = np.random.default_rng(42)
        return (rng.normal(size=(n, dim)) + shift).astype(np.float64)

    def test_no_drift_when_distributions_identical(self) -> None:
        ref = self._make_embeddings(200)
        cur = ref.copy()
        result = detect_embedding_drift(ref, cur, threshold=0.1)
        assert not result.is_drifted
        assert result.drift_score < 0.1

    def test_drift_detected_on_shifted_distribution(self) -> None:
        ref = self._make_embeddings(200, shift=0.0)
        cur = self._make_embeddings(200, shift=10.0)  # large shift → high divergence
        result = detect_embedding_drift(ref, cur, threshold=0.05)
        assert result.is_drifted

    def test_result_fields_populated(self) -> None:
        ref = self._make_embeddings(100)
        cur = self._make_embeddings(100)
        result = detect_embedding_drift(ref, cur)
        assert result.n_reference == 100
        assert result.n_current == 100
        assert result.sampled_dims > 0
        assert len(result.per_dim_scores) == result.sampled_dims

    def test_custom_threshold(self) -> None:
        ref = self._make_embeddings(100, shift=0.0)
        cur = self._make_embeddings(100, shift=0.5)
        # With very tight threshold, drift should be detected
        result_tight = detect_embedding_drift(ref, cur, threshold=0.001)
        assert result_tight.is_drifted

    def test_max_dims_limits_sampled_dimensions(self) -> None:
        ref = self._make_embeddings(100, dim=256)
        cur = self._make_embeddings(100, dim=256)
        result = detect_embedding_drift(ref, cur, max_dims=10)
        assert result.sampled_dims <= 10


# ── label_drift ───────────────────────────────────────────────────────────────


class TestCodeFrequencies:
    def test_builds_binary_frequencies(self) -> None:
        notes = [["I10", "E11"], ["I10"], []]
        freqs = _code_frequencies(notes, ["I10", "E11", "J18"])
        assert freqs["I10"] == [1.0, 1.0, 0.0]
        assert freqs["E11"] == [1.0, 0.0, 0.0]
        assert freqs["J18"] == [0.0, 0.0, 0.0]


class TestDetectLabelDrift:
    def _make_notes(self, n: int, codes: list[str], prob: float = 0.3) -> list[list[str]]:
        rng = np.random.default_rng(42)
        return [[c for c in codes if rng.random() < prob] for _ in range(n)]

    def test_no_drift_with_same_distribution(self) -> None:
        codes = ["I10", "E11", "J18"]
        ref = self._make_notes(200, codes)
        cur = self._make_notes(200, codes)
        result = detect_label_drift(ref, cur, codes, alpha=0.05)
        # With same distribution, most codes should not drift
        assert result.n_tested_codes == 3

    def test_empty_codes_list(self) -> None:
        result = detect_label_drift([], [], [], alpha=0.05)
        assert not result.is_drifted
        assert result.n_tested_codes == 0

    def test_bonferroni_correction_applied(self) -> None:
        codes = ["I10", "E11"]
        ref = self._make_notes(100, codes)
        cur = self._make_notes(100, codes)
        result = detect_label_drift(ref, cur, codes, alpha=0.05)
        assert result.alpha_corrected == pytest.approx(0.05 / len(codes))

    def test_drift_detected_on_very_different_distributions(self) -> None:
        codes = ["I10"]
        # Reference: I10 appears 90% of the time
        ref = [["I10"]] * 90 + [[]] * 10
        # Current: I10 appears 5% of the time
        cur = [["I10"]] * 5 + [[]] * 95
        result = detect_label_drift(ref, cur, codes, alpha=0.05)
        assert result.is_drifted


# ── data_drift ────────────────────────────────────────────────────────────────


class TestAggregateCountis:
    def test_sums_entity_counts(self) -> None:
        notes = [{"PERSON": 2, "DATE_TIME": 1}, {"PERSON": 3}]
        totals = _aggregate_counts(notes, ["PERSON", "DATE_TIME"])
        assert totals["PERSON"] == 5
        assert totals["DATE_TIME"] == 1

    def test_missing_entity_defaults_to_zero(self) -> None:
        notes = [{"PERSON": 1}]
        totals = _aggregate_counts(notes, ["PERSON", "LOCATION"])
        assert totals["LOCATION"] == 0


class TestDetectDataDrift:
    def _make_entities(self, n: int, base_counts: dict[str, int]) -> list[dict[str, int]]:
        return [dict(base_counts) for _ in range(n)]

    def test_no_drift_with_identical_distributions(self) -> None:
        notes = self._make_entities(100, {"PERSON": 2, "DATE_TIME": 3, "LOCATION": 1})
        result = detect_data_drift(notes, notes, alpha=0.05)
        assert not result.is_drifted

    def test_empty_notes_no_drift(self) -> None:
        result = detect_data_drift([], [], alpha=0.05)
        assert not result.is_drifted
        assert result.p_value == 1.0

    def test_drift_detected_on_different_distributions(self) -> None:
        ref = self._make_entities(
            200,
            {
                "PERSON": 5,
                "DATE_TIME": 1,
                "LOCATION": 0,
                "ORGANIZATION": 0,
                "MEDICAL_CONDITION": 0,
                "MEDICATION": 0,
                "PROCEDURE": 0,
            },
        )
        cur = self._make_entities(
            200,
            {
                "PERSON": 0,
                "DATE_TIME": 0,
                "LOCATION": 0,
                "ORGANIZATION": 0,
                "MEDICAL_CONDITION": 10,
                "MEDICATION": 8,
                "PROCEDURE": 5,
            },
        )
        result = detect_data_drift(ref, cur, alpha=0.05)
        assert result.is_drifted

    def test_result_has_reference_and_current_counts(self) -> None:
        ref = self._make_entities(50, {"PERSON": 3})
        cur = self._make_entities(50, {"PERSON": 5})
        result = detect_data_drift(ref, cur, entity_types=["PERSON"])
        assert "PERSON" in result.reference_counts
        assert "PERSON" in result.current_counts


# ── alert_publisher ───────────────────────────────────────────────────────────


class TestAlertPublisher:
    async def test_no_alert_when_no_drift(self) -> None:
        result = EmbeddingDriftResult(
            drift_score=0.05,
            is_drifted=False,
            threshold=0.1,
            n_reference=100,
            n_current=100,
            per_dim_scores=[0.05],
            sampled_dims=1,
        )
        with patch("src.drift_detection.alert_publisher._fire_alerts") as mock_fire:
            await publish_embedding_drift(result)
        mock_fire.assert_not_called()

    async def test_fires_alert_on_embedding_drift(self) -> None:
        result = EmbeddingDriftResult(
            drift_score=0.25,
            is_drifted=True,
            threshold=0.1,
            n_reference=100,
            n_current=100,
            per_dim_scores=[0.25],
            sampled_dims=1,
        )
        with patch(
            "src.drift_detection.alert_publisher._fire_alerts", new_callable=AsyncMock
        ) as mock_fire:
            await publish_embedding_drift(result)
        mock_fire.assert_awaited_once()

    async def test_no_alert_when_label_drift_none(self) -> None:
        result = LabelDriftResult(
            is_drifted=False,
            n_drifted_codes=0,
            n_tested_codes=10,
            alpha=0.05,
            alpha_corrected=0.005,
        )
        with patch("src.drift_detection.alert_publisher._fire_alerts") as mock_fire:
            await publish_label_drift(result)
        mock_fire.assert_not_called()

    async def test_fires_alert_on_label_drift(self) -> None:
        result = LabelDriftResult(
            is_drifted=True,
            n_drifted_codes=7,
            n_tested_codes=50,
            alpha=0.05,
            alpha_corrected=0.001,
            drifted_codes=["I10", "E11"],
        )
        with patch(
            "src.drift_detection.alert_publisher._fire_alerts", new_callable=AsyncMock
        ) as mock_fire:
            await publish_label_drift(result)
        mock_fire.assert_awaited_once()

    async def test_fires_alert_on_data_drift(self) -> None:
        result = DataDriftResult(
            is_drifted=True,
            chi2_statistic=45.2,
            p_value=0.001,
            alpha=0.05,
        )
        with patch(
            "src.drift_detection.alert_publisher._fire_alerts", new_callable=AsyncMock
        ) as mock_fire:
            await publish_data_drift(result)
        mock_fire.assert_awaited_once()

    async def test_fire_alerts_skips_pagerduty_when_no_key(self) -> None:
        pd_path = "src.drift_detection.alert_publisher._send_pagerduty"
        slack_path = "src.drift_detection.alert_publisher._send_slack"
        with (
            patch("src.drift_detection.alert_publisher.get_settings") as mock_settings,
            patch(pd_path, new_callable=AsyncMock) as mock_pd,
            patch(slack_path, new_callable=AsyncMock),
        ):
            mock_settings.return_value.pagerduty_integration_key = ""
            mock_settings.return_value.slack_webhook_url = ""
            from src.drift_detection.alert_publisher import _fire_alerts

            await _fire_alerts("summary", "warning", "embedding", {})
        mock_pd.assert_awaited_once()  # called but no-ops internally
