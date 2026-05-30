"""MLflow model promotion gate — evaluate Staging model and promote to Production."""

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
    """Load the latest Staging version of a registered model."""
    client = _get_mlflow_client()
    versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not versions:
        raise ValueError(f"No Staging version found for model {model_name!r}")
    model_uri = f"models:/{model_name}/Staging"
    import mlflow

    return mlflow.pyfunc.load_model(model_uri)


def _transition(model_name: str, version: str, stage: str) -> None:
    client = _get_mlflow_client()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )
    logger.info("model_stage_transition", model=model_name, version=version, stage=stage)


def promote_icd10(
    model_name: str,
    y_true: Any,
    y_pred_prob: Any,
    threshold: float = _DEFAULT_ICD10_THRESHOLD,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate Staging ICD-10 model and promote to Production if macro-F1 ≥ threshold.

    Raises ModelPromotionBlockedError if the model does not pass the quality gate.
    """
    client = _get_mlflow_client()
    versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not versions:
        raise ValueError(f"No Staging model found for {model_name!r}")
    version = versions[0].version

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

    _transition(model_name, version, "Production")
    logger.info("model_promoted", model=model_name, version=version, f1_macro=f1)
    return {"model_name": model_name, "version": version, "f1_macro": f1, "promoted": True}


def promote_triage(
    model_name: str,
    y_true: Any,
    y_pred: Any,
    threshold: float = _DEFAULT_TRIAGE_THRESHOLD,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate Staging triage model and promote to Production if weighted-F1 ≥ threshold.

    Raises ModelPromotionBlockedError if the model does not pass the quality gate.
    """
    client = _get_mlflow_client()
    versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not versions:
        raise ValueError(f"No Staging model found for {model_name!r}")
    version = versions[0].version

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

    _transition(model_name, version, "Production")
    logger.info("model_promoted", model=model_name, version=version, f1_weighted=f1)
    return {"model_name": model_name, "version": version, "f1_weighted": f1, "promoted": True}
