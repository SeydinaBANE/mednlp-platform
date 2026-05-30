"""Label drift detection using two-sample KS test with Bonferroni correction.

Compares ICD-10 code frequency distributions between a reference period and
the current production window. Bonferroni correction controls family-wise error
rate across the N code comparisons.
"""

from dataclasses import dataclass, field

import structlog
from scipy.stats import ks_2samp

logger = structlog.get_logger(__name__)


@dataclass
class LabelDriftResult:
    is_drifted: bool
    n_drifted_codes: int
    n_tested_codes: int
    alpha: float
    alpha_corrected: float
    drifted_codes: list[str] = field(default_factory=list)
    per_code_pvalue: dict[str, float] = field(default_factory=dict)
    per_code_statistic: dict[str, float] = field(default_factory=dict)


def detect_label_drift(
    reference_codes: list[list[str]],
    current_codes: list[list[str]],
    all_codes: list[str],
    alpha: float = 0.05,
) -> LabelDriftResult:
    """Detect ICD-10 label frequency drift using KS test + Bonferroni correction.

    Args:
        reference_codes: list of per-note ICD-10 code lists (reference window)
        current_codes:   list of per-note ICD-10 code lists (current window)
        all_codes:       exhaustive list of codes to test (e.g. top-50)
        alpha:           family-wise error rate (Bonferroni corrects for n_codes tests)
    """
    n_codes = len(all_codes)
    if n_codes == 0:
        return LabelDriftResult(
            is_drifted=False,
            n_drifted_codes=0,
            n_tested_codes=0,
            alpha=alpha,
            alpha_corrected=alpha,
        )

    alpha_corrected = alpha / n_codes  # Bonferroni correction

    ref_freqs = _code_frequencies(reference_codes, all_codes)
    cur_freqs = _code_frequencies(current_codes, all_codes)

    drifted: list[str] = []
    pvalues: dict[str, float] = {}
    statistics: dict[str, float] = {}

    for code in all_codes:
        ref_dist = ref_freqs[code]
        cur_dist = cur_freqs[code]

        if len(ref_dist) < 2 or len(cur_dist) < 2:
            continue

        stat, pvalue = ks_2samp(ref_dist, cur_dist)
        pvalues[code] = float(pvalue)
        statistics[code] = float(stat)

        if pvalue < alpha_corrected:
            drifted.append(code)

    is_drifted = len(drifted) > 0
    logger.info(
        "label_drift_computed",
        n_drifted=len(drifted),
        n_tested=n_codes,
        alpha_corrected=round(alpha_corrected, 6),
        drifted_codes=drifted[:5],  # log first 5
    )

    return LabelDriftResult(
        is_drifted=is_drifted,
        n_drifted_codes=len(drifted),
        n_tested_codes=n_codes,
        alpha=alpha,
        alpha_corrected=alpha_corrected,
        drifted_codes=drifted,
        per_code_pvalue=pvalues,
        per_code_statistic=statistics,
    )


def _code_frequencies(
    notes_codes: list[list[str]],
    all_codes: list[str],
) -> dict[str, list[float]]:
    """Return per-note binary frequency (0/1) for each code across all notes."""
    code_set = set(all_codes)
    freqs: dict[str, list[float]] = {c: [] for c in all_codes}

    for note_codes in notes_codes:
        present = set(note_codes) & code_set
        for code in all_codes:
            freqs[code].append(1.0 if code in present else 0.0)

    return freqs
