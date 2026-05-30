"""Evaluation metrics: macro-F1, AUC-ROC, bootstrap confidence intervals."""

from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

_N_BOOTSTRAP = 1000
_CI_ALPHA = 0.05  # 95% CI


# ── Bootstrap CI ──────────────────────────────────────────────────────────────


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = _N_BOOTSTRAP,
    alpha: float = _CI_ALPHA,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap confidence interval for a list of per-sample scores."""
    rng = np.random.default_rng(seed)
    arr = np.array(scores)
    means = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    )
    lower = float(np.percentile(means, 100 * (alpha / 2)))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lower, upper


# ── ICD-10 (multi-label) ──────────────────────────────────────────────────────


def evaluate_icd10(
    y_true: Any,
    y_pred_prob: Any,
    threshold: float = 0.5,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate multi-label ICD-10 predictions.

    Args:
        y_true: shape (n_samples, n_labels) binary ground truth
        y_pred_prob: shape (n_samples, n_labels) sigmoid probabilities
        threshold: decision threshold for binary prediction

    Returns dict with: f1_macro, auc_roc_macro, per_label_f1, ci_lower, ci_upper
    """
    from sklearn.metrics import f1_score, roc_auc_score

    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_pred_prob)
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    f1_macro = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    per_label_f1: list[float] = f1_score(
        y_true_arr, y_pred_arr, average=None, zero_division=0
    ).tolist()

    # AUC-ROC only for labels that have both classes in y_true
    auc_scores = []
    for i in range(y_true_arr.shape[1]):
        if y_true_arr[:, i].sum() > 0 and (1 - y_true_arr[:, i]).sum() > 0:
            auc_scores.append(float(roc_auc_score(y_true_arr[:, i], y_prob_arr[:, i])))

    auc_roc_macro = float(np.mean(auc_scores)) if auc_scores else 0.0
    ci_lower, ci_upper = bootstrap_ci(per_label_f1)

    result: dict[str, Any] = {
        "f1_macro": f1_macro,
        "auc_roc_macro": auc_roc_macro,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_labels": len(per_label_f1),
    }
    if label_names:
        result["per_label_f1"] = dict(zip(label_names, per_label_f1, strict=False))

    logger.info("icd10_eval_done", f1_macro=round(f1_macro, 4), auc_roc=round(auc_roc_macro, 4))
    return result


# ── Triage (multi-class) ──────────────────────────────────────────────────────


def evaluate_triage(
    y_true: Any,
    y_pred: Any,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate 5-class ESI triage predictions.

    Returns dict with: f1_weighted, f1_macro, per_class_f1, ci_lower, ci_upper
    """
    from sklearn.metrics import f1_score

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    f1_weighted = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    per_class: list[float] = f1_score(
        y_true_arr, y_pred_arr, average=None, zero_division=0
    ).tolist()

    ci_lower, ci_upper = bootstrap_ci(per_class)

    result: dict[str, Any] = {
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
    if class_names:
        result["per_class_f1"] = dict(zip(class_names, per_class, strict=False))

    logger.info(
        "triage_eval_done",
        f1_weighted=round(f1_weighted, 4),
        f1_macro=round(f1_macro, 4),
    )
    return result
