"""Model initialisation, 8-bit quantization and LoRA injection (Sections 3.2.1-3.2.3)."""

from __future__ import annotations

import logging

import torch
from transformers import AutoProcessor, BitsAndBytesConfig

from .router_patch import apply_router_patch

logger = logging.getLogger(__name__)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _model_class():
    try:
        from transformers import Llama4ForConditionalGeneration
        return Llama4ForConditionalGeneration
    except ImportError:  # older transformers: fall back to the generic class
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM


def load_processor(name_or_path: str):
    processor = AutoProcessor.from_pretrained(name_or_path)
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return processor


def load_base_model(cfg: dict):
    """Load Llama-4 Scout, optionally in 8-bit with frozen weights."""
    if cfg.get("patch_router", True):
        apply_router_patch()

    dtype = DTYPES[cfg.get("torch_dtype", "bfloat16")]
    kwargs = {"torch_dtype": dtype, "device_map": cfg.get("device_map", "auto")}
    if cfg.get("load_in_8bit", True):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=cfg.get("llm_int8_threshold", 6.0),
            # Keep the router and output head in high precision so expert
            # selection is not perturbed by quantization noise.
            llm_int8_skip_modules=["router", "lm_head", "multi_modal_projector", "vision_model"],
        )

    model = _model_class().from_pretrained(cfg["name_or_path"], **kwargs)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad = False
    return model


def add_lora(model, lora_cfg: dict, gradient_checkpointing: bool = True):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()

    config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        # Restrict adapters to the language model; the vision tower stays frozen.
        exclude_modules=r".*vision_model.*",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    trainable, total = count_parameters(model)
    logger.info("Trainable params: %s / %s (%.4f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)
    return model


def load_for_inference(cfg: dict):
    model = load_base_model(cfg)
    if cfg.get("adapter_path"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
    model.eval()
    return model


def count_parameters(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
