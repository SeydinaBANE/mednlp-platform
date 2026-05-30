"""LoRA-Mistral embedder — 4-bit quantised, loaded from MLflow registry.

Requires GPU (CUDA) and the `gpu` optional dependency group:
  uv sync --extra gpu
"""

import asyncio
from functools import lru_cache
from typing import Any

import structlog

from src.core.telemetry import EMBEDDING_INFERENCE_LATENCY
from src.embeddings.base_embedder import BaseEmbedder

logger = structlog.get_logger(__name__)

_VECTOR_SIZE = 4096
_MEAN_POOLING_LAYER = -1


@lru_cache(maxsize=1)
def _load_model_and_tokenizer(mlflow_uri: str, model_name: str) -> tuple[Any, Any]:
    """Load 4-bit quantised LoRA model from MLflow registry. Cached after first load."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "LoRA-Mistral embedder requires the 'gpu' extras: uv sync --extra gpu"
        ) from exc

    import mlflow

    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    latest = client.get_latest_versions(model_name, stages=["Production"])
    if not latest:
        raise RuntimeError(f"No Production version found for MLflow model {model_name!r}")

    model_uri = f"models:/{model_name}/Production"
    local_path = mlflow.artifacts.download_artifacts(model_uri)

    logger.info("loading_lora_mistral", model_name=model_name, path=local_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        local_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base_model, local_path)
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("lora_mistral_loaded")
    return model, tokenizer


class LoraMistralEmbedder(BaseEmbedder):
    """Embeds clinical text using a LoRA-fine-tuned Mistral 7B (4-bit quantised).

    Requires CUDA and the `gpu` extras group. Inference runs in a thread pool.
    """

    def __init__(self, mlflow_uri: str, mlflow_model_name: str, version: str = "v1") -> None:
        self._mlflow_uri = mlflow_uri
        self._mlflow_model_name = mlflow_model_name
        self._version = version

    @property
    def model_name(self) -> str:
        return "lora-mistral"

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def vector_size(self) -> int:
        return _VECTOR_SIZE

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        vectors: list[list[float]] = await loop.run_in_executor(None, self._encode_sync, texts)
        return vectors

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        import time

        import torch

        model, tokenizer = _load_model_and_tokenizer(self._mlflow_uri, self._mlflow_model_name)
        start = time.perf_counter()

        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Mean-pool the last hidden state (ignore padding tokens)
        hidden = outputs.hidden_states[_MEAN_POOLING_LAYER]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        vectors = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        vectors = torch.nn.functional.normalize(vectors, p=2, dim=-1)

        elapsed = time.perf_counter() - start
        EMBEDDING_INFERENCE_LATENCY.labels(model=self.model_name).observe(elapsed)

        result: list[list[float]] = vectors.cpu().float().tolist()
        return result
