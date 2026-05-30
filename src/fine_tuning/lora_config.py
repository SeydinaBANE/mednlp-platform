"""PEFT LoRA configuration factory for ICD-10, triage, and embedding fine-tuning."""

from dataclasses import dataclass, field
from typing import Any, Literal

_DEFAULT_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class LoraHyperparams:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str] = field(default_factory=lambda: list(_DEFAULT_TARGET_MODULES))


def get_peft_config(task: str, hp: LoraHyperparams | None = None) -> Any:
    """Return a PEFT LoraConfig for the given task.

    task: "icd10" | "triage" | "embedding"
    """
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise ImportError("peft is required: uv sync --extra gpu") from exc

    hp = hp or LoraHyperparams()

    task_type_map = {
        "icd10": TaskType.SEQ_CLS,
        "triage": TaskType.SEQ_CLS,
        "embedding": TaskType.CAUSAL_LM,
    }
    if task not in task_type_map:
        raise ValueError(f"Unknown task {task!r}. Expected one of: {list(task_type_map)}")

    return LoraConfig(
        r=hp.r,
        lora_alpha=hp.lora_alpha,
        lora_dropout=hp.lora_dropout,
        bias=hp.bias,
        task_type=task_type_map[task],
        target_modules=hp.target_modules,
        inference_mode=False,
    )


def get_icd10_config(hp: LoraHyperparams | None = None) -> Any:
    return get_peft_config("icd10", hp)


def get_triage_config(hp: LoraHyperparams | None = None) -> Any:
    return get_peft_config("triage", hp)


def get_embedding_config(hp: LoraHyperparams | None = None) -> Any:
    return get_peft_config("embedding", hp)
