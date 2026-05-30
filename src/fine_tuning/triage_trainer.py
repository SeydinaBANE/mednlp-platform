"""Triage (ESI) 5-class LoRA fine-tuner — weighted CrossEntropyLoss."""

from dataclasses import dataclass, field
from typing import Any

import structlog

from src.fine_tuning.lora_config import LoraHyperparams, get_triage_config

logger = structlog.get_logger(__name__)

_DEFAULT_BASE_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
_NUM_CLASSES = 5  # ESI levels 1-5 → classes 0-4
_DEFAULT_OUTPUT_DIR = "artifacts/triage"

# ESI class weights: ESI-1 (critical) over-represented in loss to avoid ignoring rare cases
_DEFAULT_CLASS_WEIGHTS = [4.0, 2.0, 1.0, 1.0, 1.0]  # ESI 1→4, 2→2, 3-5→1


@dataclass
class TriageTrainingArgs:
    output_dir: str = _DEFAULT_OUTPUT_DIR
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = True
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1_weighted"
    logging_steps: int = 50
    class_weights: list[float] = field(default_factory=lambda: list(_DEFAULT_CLASS_WEIGHTS))
    lora: LoraHyperparams = field(default_factory=LoraHyperparams)


def train_triage(
    base_model_name: str = _DEFAULT_BASE_MODEL,
    train_dataset: Any = None,
    eval_dataset: Any = None,
    args: TriageTrainingArgs | None = None,
    mlflow_experiment: str = "triage-lora",
) -> dict[str, Any]:
    """Fine-tune BiomedBERT with LoRA for 5-class ESI triage classification.

    Uses class-weighted CrossEntropyLoss to handle ESI-1/2 under-representation.
    Returns a dict with: model_path, best_f1_weighted, training_loss.
    Requires GPU extras: uv sync --extra gpu
    """
    try:
        import mlflow
        import torch
        from peft import get_peft_model
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError("GPU extras required: uv sync --extra gpu") from exc

    cfg = args or TriageTrainingArgs()
    lora_config = get_triage_config(cfg.lora)

    logger.info("triage_training_started", base_model=base_model_name)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)  # type: ignore[no-untyped-call]
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=_NUM_CLASSES
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    class_weights_tensor = torch.tensor(cfg.class_weights, dtype=torch.float)

    class _WeightedTrainer(Trainer):
        def compute_loss(  # type: ignore[override]
            self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any
        ) -> Any:
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    def _tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=512
        )
        return result

    if train_dataset is not None:
        train_dataset = train_dataset.map(_tokenize, batched=True)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(_tokenize, batched=True)

    def _compute_metrics(eval_pred: Any) -> dict[str, float]:
        from sklearn.metrics import f1_score

        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
        f1_macro = float(f1_score(labels, preds, average="macro", zero_division=0))
        return {"f1_weighted": f1, "f1_macro": f1_macro}

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        fp16=cfg.fp16,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        logging_steps=cfg.logging_steps,
        report_to="mlflow",
    )

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run():
        trainer = _WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=_compute_metrics,
        )
        train_result = trainer.train()
        trainer.save_model(cfg.output_dir)

        best_f1 = trainer.state.best_metric or 0.0
        mlflow.log_metrics({"best_f1_weighted": best_f1})
        mlflow.log_param("base_model", base_model_name)
        mlflow.log_param("class_weights", cfg.class_weights)

    logger.info("triage_training_done", output_dir=cfg.output_dir, best_f1=best_f1)
    return {
        "model_path": cfg.output_dir,
        "best_f1_weighted": best_f1,
        "training_loss": train_result.training_loss,
    }
