"""MLflow model registry client — fetch model metadata and artifacts."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from src.core.config import get_settings

if TYPE_CHECKING:
    import mlflow.tracking

logger = structlog.get_logger(__name__)

_client: Any = None


@dataclass
class ModelInfo:
    name: str
    version: str
    stage: str
    run_id: str
    artifact_uri: str


def _get_mlflow_client() -> "mlflow.tracking.MlflowClient":
    global _client
    if _client is None:
        import mlflow

        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        _client = mlflow.tracking.MlflowClient()
    return _client  # type: ignore[no-any-return]


def get_production_model(model_name: str) -> ModelInfo:
    """Return metadata for the model aliased as 'champion' (MLflow 3.x production alias).

    Falls back to 'production' alias for backward compatibility.
    """
    client = _get_mlflow_client()
    import mlflow.exceptions

    for alias in ("champion", "production"):
        try:
            mv = client.get_model_version_by_alias(model_name, alias)
            return ModelInfo(
                name=mv.name,
                version=mv.version,
                stage=alias,
                run_id=mv.run_id,
                artifact_uri=mv.source,
            )
        except mlflow.exceptions.MlflowException:
            continue

    raise ValueError(
        f"No production alias ('champion' or 'production') found for model {model_name!r}"
    )


def get_model_by_version(model_name: str, version: str) -> ModelInfo:
    """Return metadata for a specific registered model version."""
    client = _get_mlflow_client()
    import mlflow

    mv: mlflow.entities.model_registry.ModelVersion = client.get_model_version(model_name, version)  # type: ignore[attr-defined]
    return ModelInfo(
        name=mv.name,
        version=mv.version,
        stage=", ".join(mv.aliases) if mv.aliases else "",
        run_id=mv.run_id,
        artifact_uri=mv.source,
    )


def list_registered_models() -> list[str]:
    """Return names of all registered models."""
    client = _get_mlflow_client()
    import mlflow

    models: list[mlflow.entities.model_registry.RegisteredModel] = list(  # type: ignore[attr-defined]
        client.search_registered_models()
    )
    return [m.name for m in models]
