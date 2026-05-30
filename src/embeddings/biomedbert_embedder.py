"""BiomedBERT embedder using sentence-transformers (CPU/GPU)."""

import asyncio
from functools import lru_cache
from typing import Any

import structlog

from src.core.telemetry import EMBEDDING_INFERENCE_LATENCY
from src.embeddings.base_embedder import BaseEmbedder

logger = structlog.get_logger(__name__)

_MODEL_ID = "pritamdeka/S-PubMedBert-MS-MARCO"
_VECTOR_SIZE = 768
_MODEL_VERSION = "v1"


@lru_cache(maxsize=1)
def _load_model() -> Any:  # noqa: ANN401
    from sentence_transformers import SentenceTransformer  # lazy import — heavy dependency

    logger.info("loading_biomedbert", model_id=_MODEL_ID)
    model = SentenceTransformer(_MODEL_ID)
    logger.info("biomedbert_loaded")
    return model


class BiomedBertEmbedder(BaseEmbedder):
    """Embeds clinical text with a PubMed-fine-tuned BERT model.

    Inference runs in a thread pool to avoid blocking the asyncio event loop.
    """

    @property
    def model_name(self) -> str:
        return "biomedbert"

    @property
    def model_version(self) -> str:
        return _MODEL_VERSION

    @property
    def vector_size(self) -> int:
        return _VECTOR_SIZE

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        vectors: list[list[float]] = await loop.run_in_executor(None, self._encode_sync, texts)
        EMBEDDING_INFERENCE_LATENCY.labels(model=self.model_name).observe(0)
        return vectors

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        import time

        model = _load_model()
        start = time.perf_counter()
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        elapsed = time.perf_counter() - start
        EMBEDDING_INFERENCE_LATENCY.labels(model=self.model_name).observe(elapsed)
        result: list[list[float]] = embeddings.tolist()
        return result
