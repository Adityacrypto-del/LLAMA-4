"""PEFT-QLoRA training (Section 3.2, Algorithm 2).

Usage:
    python -m qvaidya.train --config configs/train_config.yaml
"""

from __future__ import annotations

import argparse
import gc
import logging
import os

import torch
import yaml
from transformers import Trainer, TrainerCallback, TrainingArguments

from .data import InstructionMaskingCollator, MedicalSFTDataset, load_sources, save_records, stratified_split
from .model import add_lora, load_base_model, load_processor

logger = logging.getLogger(__name__)


class MemoryCleanupCallback(TrainerCallback):
    """Explicit garbage collection after every epoch to limit fragmentation."""

    def on_epoch_end(self, args, state, control, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_training_args(cfg: dict) -> TrainingArguments:
    t = cfg["training"]
    return TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        max_grad_norm=t["max_grad_norm"],
        optim=t["optim"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=t["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=t.get("report_to", "none"),
        seed=cfg["data"].get("seed", 42),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Q-VAIDYA FP8-QLoRA fine-tuning")
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Phase 1: data preparation
    records = load_sources(cfg["data"]["sources"])
    train_records, val_records = stratified_split(records, cfg["data"]["val_ratio"], cfg["data"]["seed"])
    logger.info("Train: %d  Val: %d", len(train_records), len(val_records))
    out_dir = cfg["training"]["output_dir"]
    save_records(val_records, os.path.join(out_dir, "val_records.jsonl"))

    # Phase 2: quantized model + adapters
    processor = load_processor(cfg["model"]["name_or_path"])
    model = load_base_model(cfg["model"])
    model = add_lora(model, cfg["lora"], cfg["training"]["gradient_checkpointing"])

    trainer = Trainer(
        model=model,
        args=build_training_args(cfg),
        train_dataset=MedicalSFTDataset(train_records),
        eval_dataset=MedicalSFTDataset(val_records),
        data_collator=InstructionMaskingCollator(processor, cfg["data"]["max_length"]),
        callbacks=[MemoryCleanupCallback()],
    )
    trainer.train()

    best_dir = os.path.join(out_dir, "best_adapter")
    trainer.model.save_pretrained(best_dir)
    processor.save_pretrained(best_dir)
    logger.info("Saved best adapter to %s", best_dir)


if __name__ == "__main__":
    main()
