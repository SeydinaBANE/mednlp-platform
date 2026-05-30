"""MLflow model promotion gate — evaluate challenger model and promote to champion."""

from typing import Any

import structlog

from src.core.exceptions import ModelPromotionBlockedError
from src.fine_tuning.evaluator import evaluate_icd10, evaluate_triage

logger = structlog.get_logger(__name__)

_DEFAULT_ICD10_THRESHOLD = 0.80  # macro-F1
_DEFAULT_TRIAGE_THRESHOLD = 0.80  # weighted-F1


def _get_mlflow_client() -> Any:
    import mlflow

    return mlflow.tracking.MlflowClient()


def _load_staging_model(model_name: str) -> Any:
    """Load the model aliased as 'challenger' (MLflow 3.x staging alias)."""
    import mlflow
    import mlflow.exceptions

    client = _get_mlflow_client()
    for alias in ("challenger", "staging"):
        try:
            client.get_model_version_by_alias(model_name, alias)
            return mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
        except mlflow.exceptions.MlflowException:
            continue

    raise ValueError(f"No staging alias ('challenger' or 'staging') found for model {model_name!r}")


def _get_challenger_version(client: Any, model_name: str) -> str:
    """Return the version number of the current challenger (staging) model."""
    import mlflow.exceptions

    for alias in ("challenger", "staging"):
        try:
            mv = client.get_model_version_by_alias(model_name, alias)
            return str(mv.version)
        except mlflow.exceptions.MlflowException:
            continue

    raise ValueError(f"No staging alias found for model {model_name!r}")


def _promote(model_name: str, version: str) -> None:
    """Set the 'champion' alias on the given version (MLflow 3.x production promotion)."""
    client = _get_mlflow_client()
    client.set_registered_model_alias(name=model_name, alias="champion", version=version)
    logger.info("model_promoted_to_champion", model=model_name, version=version)


def promote_icd10(
    model_name: str,
    y_true: Any,
    y_pred_prob: Any,
    threshold: float = _DEFAULT_ICD10_THRESHOLD,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate challenger ICD-10 model and promote to champion if macro-F1 ≥ threshold.

    Raises ModelPromotionBlockedError if the model does not pass the quality gate.
    """
    client = _get_mlflow_client()
    version = _get_challenger_version(client, model_name)

    metrics = evaluate_icd10(y_true, y_pred_prob, label_names=label_names)
    f1 = metrics["f1_macro"]

    logger.info("promotion_gate_icd10", model=model_name, f1_macro=f1, threshold=threshold)

    if f1 < threshold:
        raise ModelPromotionBlockedError(
            model_name=f"{model_name}@{version}",
            metric="f1_macro",
            actual=f1,
            threshold=threshold,
        )

    _promote(model_name, version)
    logger.info("model_promoted", model=model_name, version=version, f1_macro=f1)
    return {"model_name": model_name, "version": version, "f1_macro": f1, "promoted": True}


def promote_triage(
    model_name: str,
    y_true: Any,
    y_pred: Any,
    threshold: float = _DEFAULT_TRIAGE_THRESHOLD,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate challenger triage model and promote to champion if weighted-F1 ≥ threshold.

    Raises ModelPromotionBlockedError if the model does not pass the quality gate.
    """
    client = _get_mlflow_client()
    version = _get_challenger_version(client, model_name)

    metrics = evaluate_triage(y_true, y_pred, class_names=class_names)
    f1 = metrics["f1_weighted"]

    logger.info("promotion_gate_triage", model=model_name, f1_weighted=f1, threshold=threshold)

    if f1 < threshold:
        raise ModelPromotionBlockedError(
            model_name=f"{model_name}@{version}",
            metric="f1_weighted",
            actual=f1,
            threshold=threshold,
        )

    _promote(model_name, version)
    logger.info("model_promoted", model=model_name, version=version, f1_weighted=f1)
    return {"model_name": model_name, "version": version, "f1_weighted": f1, "promoted": True}
