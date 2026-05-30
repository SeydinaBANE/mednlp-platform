"""Embedding drift detection using Jensen-Shannon divergence.

Computes JS divergence between a reference embedding distribution and the current
production distribution. Alerts when divergence exceeds THRESHOLD (default 0.1).
"""

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray
from scipy.stats import entropy

from src.core.config import get_settings
from src.core.telemetry import DLQ_MESSAGES_TOTAL

logger = structlog.get_logger(__name__)

_HISTOGRAM_BINS = 50
_EPSILON = 1e-10


@dataclass
class EmbeddingDriftResult:
    drift_score: float  # mean JS divergence across dimensions
    is_drifted: bool
    threshold: float
    n_reference: int
    n_current: int
    per_dim_scores: list[float]  # JS divergence per sampled dimension
    sampled_dims: int


def js_divergence(
    p: NDArray[np.float64], q: NDArray[np.float64], bins: int = _HISTOGRAM_BINS
) -> float:
    """Jensen-Shannon divergence between two 1-D distributions.

    Returns a value in [0, ln(2)] ≈ [0, 0.693].
    A score > 0.1 indicates meaningful drift.
    """
    min_val = float(min(p.min(), q.min()))
    max_val = float(max(p.max(), q.max()))
    if max_val == min_val:
        return 0.0

    p_hist, _ = np.histogram(p, bins=bins, range=(min_val, max_val))
    q_hist, _ = np.histogram(q, bins=bins, range=(min_val, max_val))

    p_hist = p_hist.astype(float) + _EPSILON
    q_hist = q_hist.astype(float) + _EPSILON
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()

    m = (p_hist + q_hist) / 2.0
    return float((entropy(p_hist, m) + entropy(q_hist, m)) / 2.0)


def detect_embedding_drift(
    reference: NDArray[np.float64],
    current: NDArray[np.float64],
    threshold: float | None = None,
    max_dims: int = 64,
) -> EmbeddingDriftResult:
    """Detect embedding drift between reference and current batches.

    Args:
        reference: shape (n_ref, embedding_dim)
        current:   shape (n_cur, embedding_dim)
        threshold: JS divergence threshold (defaults to settings.embedding_drift_threshold)
        max_dims:  number of dimensions to sample (avoids O(dim) overhead for large models)

    Returns an EmbeddingDriftResult with the mean JS divergence score.
    """
    if threshold is None:
        threshold = get_settings().embedding_drift_threshold

    n_ref, dim = reference.shape
    n_cur = current.shape[0]

    # Sample dimensions evenly to keep computation bounded
    sampled_dims = min(dim, max_dims)
    step = max(1, dim // sampled_dims)
    dim_indices = list(range(0, dim, step))[:sampled_dims]

    per_dim: list[float] = []
    for d in dim_indices:
        score = js_divergence(reference[:, d], current[:, d])
        per_dim.append(score)

    drift_score = float(np.mean(per_dim))
    is_drifted = drift_score > threshold

    logger.info(
        "embedding_drift_computed",
        drift_score=round(drift_score, 4),
        threshold=threshold,
        is_drifted=is_drifted,
        n_reference=n_ref,
        n_current=n_cur,
        sampled_dims=len(dim_indices),
    )

    if is_drifted:
        DLQ_MESSAGES_TOTAL.labels(reason="embedding_drift").inc()

    return EmbeddingDriftResult(
        drift_score=drift_score,
        is_drifted=is_drifted,
        threshold=threshold,
        n_reference=n_ref,
        n_current=n_cur,
        per_dim_scores=per_dim,
        sampled_dims=len(dim_indices),
    )
