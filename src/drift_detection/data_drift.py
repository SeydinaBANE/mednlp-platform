"""Data (NER entity) drift detection using chi-squared test on entity count distributions."""

from dataclasses import dataclass, field

import numpy as np
import structlog
from scipy.stats import chi2_contingency

logger = structlog.get_logger(__name__)

# Entity types tracked from presidio/spacy NER pipeline
_DEFAULT_ENTITY_TYPES = [
    "PERSON",
    "DATE_TIME",
    "LOCATION",
    "ORGANIZATION",
    "MEDICAL_CONDITION",
    "MEDICATION",
    "PROCEDURE",
]


@dataclass
class DataDriftResult:
    is_drifted: bool
    chi2_statistic: float
    p_value: float
    alpha: float
    reference_counts: dict[str, int] = field(default_factory=dict)
    current_counts: dict[str, int] = field(default_factory=dict)
    entity_types: list[str] = field(default_factory=list)


def detect_data_drift(
    reference_entities: list[dict[str, int]],
    current_entities: list[dict[str, int]],
    entity_types: list[str] | None = None,
    alpha: float = 0.05,
) -> DataDriftResult:
    """Detect NER entity count distribution drift using chi-squared test.

    Args:
        reference_entities: list of per-note entity count dicts (reference window)
        current_entities:   list of per-note entity count dicts (current window)
        entity_types:       entity types to compare (defaults to _DEFAULT_ENTITY_TYPES)
        alpha:              significance level for chi-squared test
    """
    entity_types = entity_types or _DEFAULT_ENTITY_TYPES

    ref_counts = _aggregate_counts(reference_entities, entity_types)
    cur_counts = _aggregate_counts(current_entities, entity_types)

    ref_vec = np.array([ref_counts[e] for e in entity_types], dtype=float)
    cur_vec = np.array([cur_counts[e] for e in entity_types], dtype=float)

    # Chi-squared test requires non-zero totals
    if ref_vec.sum() == 0 or cur_vec.sum() == 0:
        logger.warning("data_drift_empty_counts")
        return DataDriftResult(
            is_drifted=False,
            chi2_statistic=0.0,
            p_value=1.0,
            alpha=alpha,
            reference_counts=ref_counts,
            current_counts=cur_counts,
            entity_types=entity_types,
        )

    # Normalise to comparable scales (per 1000 notes)
    n_ref = max(len(reference_entities), 1)
    n_cur = max(len(current_entities), 1)
    ref_norm = (ref_vec / n_ref * 1000).astype(int) + 1
    cur_norm = (cur_vec / n_cur * 1000).astype(int) + 1

    contingency = np.array([ref_norm, cur_norm])
    chi2, p_value, *_ = chi2_contingency(contingency)
    chi2 = float(chi2)
    p_value = float(p_value)

    is_drifted = p_value < alpha
    logger.info(
        "data_drift_computed",
        chi2=round(chi2, 4),
        p_value=round(p_value, 6),
        is_drifted=is_drifted,
        n_reference=n_ref,
        n_current=n_cur,
    )

    return DataDriftResult(
        is_drifted=is_drifted,
        chi2_statistic=chi2,
        p_value=p_value,
        alpha=alpha,
        reference_counts=ref_counts,
        current_counts=cur_counts,
        entity_types=entity_types,
    )


def _aggregate_counts(
    notes: list[dict[str, int]],
    entity_types: list[str],
) -> dict[str, int]:
    totals: dict[str, int] = {e: 0 for e in entity_types}
    for note in notes:
        for entity in entity_types:
            totals[entity] += note.get(entity, 0)
    return totals
