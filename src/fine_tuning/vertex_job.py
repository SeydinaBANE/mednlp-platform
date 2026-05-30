"""Submit fine-tuning jobs to Vertex AI Custom Training."""

from dataclasses import dataclass

import structlog

from src.core.config import get_settings

logger = structlog.get_logger(__name__)

_MACHINE_TYPE = "n1-standard-8"
_ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
_ACCELERATOR_COUNT = 1
_CONTAINER_URI = "gcr.io/{project}/mednlp-fine-tune:latest"
_REPLICA_COUNT = 1

_VALID_TASKS = ("icd10", "triage")


@dataclass
class VertexJobConfig:
    task: str  # "icd10" | "triage"
    display_name: str = ""
    machine_type: str = _MACHINE_TYPE
    accelerator_type: str = _ACCELERATOR_TYPE
    accelerator_count: int = _ACCELERATOR_COUNT
    base_output_dir: str = ""  # GCS URI — defaults to gcs_bucket_artifacts/fine_tuning/
    sync: bool = False  # wait for completion


def submit_fine_tune_job(config: VertexJobConfig) -> str:
    """Submit a LoRA fine-tuning job to Vertex AI Custom Training.

    Returns the Vertex AI job resource name (e.g. projects/.../trainingPipelines/...).
    Requires `google-cloud-aiplatform` and GCP credentials.
    """
    try:
        from google.cloud import aiplatform
    except ImportError as exc:
        raise ImportError("google-cloud-aiplatform is required for Vertex jobs") from exc

    if config.task not in _VALID_TASKS:
        raise ValueError(f"task must be one of {_VALID_TASKS}, got {config.task!r}")

    settings = get_settings()
    aiplatform.init(project=settings.gcp_project_id, location=settings.gcp_region)

    display_name = config.display_name or f"mednlp-{config.task}-lora-finetune"
    base_output = config.base_output_dir or (
        f"gs://{settings.gcs_bucket_artifacts}/fine_tuning/{config.task}"
    )
    container_uri = _CONTAINER_URI.format(project=settings.gcp_project_id)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=container_uri,
        command=["python", "-m", f"src.fine_tuning.{config.task}_trainer"],
    )

    job.run(
        machine_type=config.machine_type,
        accelerator_type=config.accelerator_type,
        accelerator_count=config.accelerator_count,
        replica_count=_REPLICA_COUNT,
        base_output_dir=base_output,
        sync=config.sync,
        environment_variables={
            "TASK": config.task,
            "GCS_BUCKET_ARTIFACTS": settings.gcs_bucket_artifacts,
        },
    )

    job_name: str = job.resource_name
    logger.info(
        "vertex_job_submitted",
        task=config.task,
        display_name=display_name,
        job_name=job_name,
        sync=config.sync,
    )
    return job_name


def main() -> None:
    """Entry point for Vertex AI training container — dispatches to trainer."""
    import os
    import sys

    task = os.environ.get("TASK", "")
    if task not in _VALID_TASKS:
        print(f"ERROR: TASK env var must be one of {_VALID_TASKS}, got {task!r}", file=sys.stderr)
        sys.exit(1)

    logger.info("vertex_container_started", task=task)

    if task == "icd10":
        from src.fine_tuning.icd10_trainer import train_icd10

        result = train_icd10()
    else:
        from src.fine_tuning.triage_trainer import train_triage

        result = train_triage()

    logger.info("vertex_container_finished", task=task, result=result)


if __name__ == "__main__":
    main()
