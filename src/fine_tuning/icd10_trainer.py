"""ICD-10 multi-label LoRA fine-tuner — BCEWithLogitsLoss, top-50 codes."""

from dataclasses import dataclass, field
from typing import Any

import structlog

from src.fine_tuning.lora_config import LoraHyperparams, get_icd10_config

logger = structlog.get_logger(__name__)

_DEFAULT_BASE_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
_DEFAULT_NUM_LABELS = 50
_DEFAULT_OUTPUT_DIR = "artifacts/icd10"


@dataclass
class ICD10TrainingArgs:
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
    metric_for_best_model: str = "eval_f1_macro"
    logging_steps: int = 50
    dataloader_num_workers: int = 4
    lora: LoraHyperparams = field(default_factory=LoraHyperparams)


def train_icd10(
    base_model_name: str = _DEFAULT_BASE_MODEL,
    num_labels: int = _DEFAULT_NUM_LABELS,
    train_dataset: Any = None,
    eval_dataset: Any = None,
    args: ICD10TrainingArgs | None = None,
    mlflow_experiment: str = "icd10-lora",
) -> dict[str, Any]:
    """Fine-tune BiomedBERT with LoRA for multi-label ICD-10 classification.

    Returns a dict with: model_path, best_f1, training_loss.
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

    cfg = args or ICD10TrainingArgs()
    lora_config = get_icd10_config(cfg.lora)

    logger.info("icd10_training_started", base_model=base_model_name, num_labels=num_labels)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

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
        preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy()
        f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
        return {"f1_macro": f1}

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
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to="mlflow",
    )

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run():
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=_compute_metrics,
        )
        train_result = trainer.train()
        trainer.save_model(cfg.output_dir)

        best_f1 = trainer.state.best_metric or 0.0
        mlflow.log_metrics({"best_f1_macro": best_f1})
        mlflow.log_param("num_labels", num_labels)
        mlflow.log_param("base_model", base_model_name)

    logger.info("icd10_training_done", output_dir=cfg.output_dir, best_f1=best_f1)
    return {
        "model_path": cfg.output_dir,
        "best_f1": best_f1,
        "training_loss": train_result.training_loss,
    }
