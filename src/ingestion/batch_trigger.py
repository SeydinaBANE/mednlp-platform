"""GCS object.finalize event handler → publish to Pub/Sub for batch ingestion.

Triggered by a Cloud Function or Cloud Run when a new file lands in GCS.
Supports .txt, .json (FHIR bundle), and .ndjson formats.
"""

import json
from typing import Any

import structlog
from google.cloud import pubsub_v1, storage

from src.core.config import get_settings
from src.core.exceptions import IngestionError

logger = structlog.get_logger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".json", ".ndjson"}


def _parse_gcs_event(event: dict[str, Any]) -> tuple[str, str]:
    """Extract bucket and object name from a GCS event payload."""
    bucket = event.get("bucket", "")
    name = event.get("name", "")
    if not bucket or not name:
        raise IngestionError(f"Invalid GCS event: missing bucket or name — {event!r}")
    return bucket, name


def _is_supported(object_name: str) -> bool:
    import os

    _, ext = os.path.splitext(object_name.lower())
    return ext in SUPPORTED_EXTENSIONS


def _read_gcs_object(bucket_name: str, object_name: str) -> bytes:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    data: bytes = blob.download_as_bytes()
    return data


def _iter_fhir_resources(content: bytes, object_name: str) -> list[dict[str, Any]]:
    """Parse content into a flat list of FHIR resource dicts."""
    import os

    _, ext = os.path.splitext(object_name.lower())

    if ext == ".ndjson":
        resources = []
        for line in content.decode("utf-8").splitlines():
            line = line.strip()
            if line:
                resources.append(json.loads(line))
        return resources

    parsed = json.loads(content)

    if isinstance(parsed, dict) and parsed.get("resourceType") == "Bundle":
        return [entry["resource"] for entry in parsed.get("entry", []) if "resource" in entry]

    if isinstance(parsed, dict):
        return [parsed]

    if isinstance(parsed, list):
        return parsed

    raise IngestionError(f"Unrecognised JSON structure in {object_name!r}")


def _publish_batch(resources: list[dict[str, Any]], source_object: str) -> int:
    """Publish each resource as an individual Pub/Sub message. Returns published count."""
    settings = get_settings()
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(settings.gcp_project_id, settings.pubsub_topic_notes)

    futures = []
    for resource in resources:
        data = json.dumps(resource).encode("utf-8")
        attrs = {"source": "gcs_batch", "gcs_object": source_object}
        futures.append(publisher.publish(topic_path, data, **attrs))

    published = 0
    for future in futures:
        try:
            future.result(timeout=10.0)
            published += 1
        except Exception as exc:
            logger.error("batch_publish_failed", error=str(exc))

    return published


def handle_gcs_event(event: dict[str, Any]) -> dict[str, Any]:
    """Entry point for a GCS object.finalize Cloud Function trigger.

    Returns a summary dict suitable for a Cloud Function response.
    """
    bucket_name, object_name = _parse_gcs_event(event)

    if not _is_supported(object_name):
        logger.info("batch_trigger_skipped", object=object_name, reason="unsupported extension")
        return {"status": "skipped", "object": object_name}

    logger.info("batch_trigger_started", bucket=bucket_name, object=object_name)

    content = _read_gcs_object(bucket_name, object_name)

    try:
        resources = _iter_fhir_resources(content, object_name)
    except (json.JSONDecodeError, IngestionError) as exc:
        logger.error("batch_parse_failed", object=object_name, error=str(exc))
        return {"status": "error", "object": object_name, "error": str(exc)}

    published = _publish_batch(resources, object_name)

    logger.info(
        "batch_trigger_completed",
        object=object_name,
        total=len(resources),
        published=published,
    )
    return {"status": "ok", "object": object_name, "total": len(resources), "published": published}
