"""Data preparation and preprocessing (Section 3.1, Algorithm 1).

Every source is converted into a common record:

    {"source": str, "label": str, "question": str, "context": str,
     "answer": str, "image": PIL.Image | None}

Records are rendered with the Llama-4 chat template, tokenized to at most
``max_length`` tokens and the labels of every prompt / padding position are set
to -100 so the causal loss only covers the assistant response.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from typing import Any

import torch
from datasets import Dataset, load_dataset, load_from_disk
from PIL import Image

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

SYSTEM_PROMPT = (
    "You are Q-VAIDYA, a clinical decision support assistant. "
    "Answer medical questions accurately and concisely."
)

PUBMEDQA_LABELS = ["yes", "no", "maybe"]
MEDQA_LABELS = ["A", "B", "C", "D"]


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def pubmedqa_prompt(question: str, context: str) -> str:
    return (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer with exactly one word: yes, no, or maybe."
    )


def medqa_prompt(question: str, options: dict[str, str]) -> str:
    opts = "\n".join(f"{k}. {options[k]}" for k in MEDQA_LABELS if k in options)
    return (
        f"Question: {question}\n\nOptions:\n{opts}\n\n"
        "Answer with the letter of the correct option (A, B, C, or D)."
    )


def open_prompt(question: str, context: str = "") -> str:
    if context:
        return f"Context:\n{context}\n\nQuestion: {question}"
    return question


def build_messages(user_text: str, has_image: bool, answer: str | None = None) -> list[dict]:
    user_content: list[dict[str, Any]] = []
    if has_image:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": user_text})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


# --------------------------------------------------------------------------- #
# Stage 1 + 2: loading and extraction
# --------------------------------------------------------------------------- #
def _take(ds: Dataset, n: int | None) -> Dataset:
    if n is None or n >= len(ds):
        return ds
    return ds.select(range(n))


def load_pubmedqa(cfg: dict) -> list[dict]:
    ds = load_dataset(cfg["hf_id"], cfg.get("config", "pqa_labeled"), split=cfg.get("split", "train"))
    ds = _take(ds, cfg.get("num_samples"))
    records = []
    for s in ds:
        context = " ".join(s["context"]["contexts"])
        label = s["final_decision"].strip().lower()
        records.append({
            "source": "pubmedqa",
            "label": label,
            "question": pubmedqa_prompt(s["question"], context),
            "context": context,
            "answer": label,
            "image": None,
        })
    return records


def load_medqa(cfg: dict) -> list[dict]:
    ds = load_dataset(cfg["hf_id"], split=cfg.get("split", "train"))
    ds = _take(ds, cfg.get("num_samples"))
    records = []
    for s in ds:
        label = s["answer_idx"].strip().upper()
        records.append({
            "source": "medqa",
            "label": label,
            "question": medqa_prompt(s["question"], s["options"]),
            "context": "",
            "answer": label,
            "image": None,
        })
    return records


def _load_any(path: str, split: str = "train") -> Dataset:
    """Load a dataset from a saved HF folder, a JSON/JSONL file, or the Hub."""
    if os.path.isdir(path):
        try:
            loaded = load_from_disk(path)
            return loaded[split] if hasattr(loaded, "keys") else loaded
        except FileNotFoundError:
            return load_dataset(path, split=split)
    if path.endswith((".json", ".jsonl")):
        return load_dataset("json", data_files=path, split="train")
    return load_dataset(path, split=split)


def _to_image(value: Any, base_dir: str | None = None) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and value.get("path"):
        value = value["path"]
    if isinstance(value, str):
        path = value if os.path.isabs(value) or base_dir is None else os.path.join(base_dir, value)
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
        logger.warning("Image not found: %s", path)
    return None


def load_awesome_medical(cfg: dict) -> list[dict]:
    path = cfg["path"]
    if not os.path.exists(path) and "/" not in path:
        logger.warning("Awesome Medical dataset not found at %s, skipping.", path)
        return []
    ds = _take(_load_any(path, cfg.get("split", "train")), cfg.get("num_samples"))
    base_dir = path if os.path.isdir(path) else os.path.dirname(path)
    q_f, a_f, i_f = cfg.get("question_field", "question"), cfg.get("answer_field", "answer"), cfg.get("image_field", "image")
    records = []
    for s in ds:
        records.append({
            "source": "awesome_medical",
            "label": "open",
            "question": open_prompt(s[q_f], s.get("context", "") or ""),
            "context": s.get("context", "") or "",
            "answer": str(s[a_f]).strip(),
            "image": _to_image(s.get(i_f), base_dir),
        })
    return records


LOADERS = {
    "pubmedqa": load_pubmedqa,
    "medqa": load_medqa,
    "awesome_medical": load_awesome_medical,
}


def load_sources(sources_cfg: dict) -> list[dict]:
    records: list[dict] = []
    for name, cfg in sources_cfg.items():
        if not cfg.get("enabled", True):
            continue
        part = LOADERS[name](cfg)
        logger.info("Loaded %d samples from %s", len(part), name)
        records.extend(part)
    return records


# --------------------------------------------------------------------------- #
# Stage 4: stratified train / validation split
# --------------------------------------------------------------------------- #
def stratified_split(records: list[dict], val_ratio: float = 0.05, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """95/5 split stratified on (source, label) so answer distributions match."""
    from sklearn.model_selection import train_test_split

    keys = [f"{r['source']}::{r['label']}" for r in records]
    counts = Counter(keys)
    # Strata with a single sample cannot be split; fold them into a shared bucket.
    keys = [k if counts[k] > 1 else "rare" for k in keys]
    idx = list(range(len(records)))
    try:
        train_idx, val_idx = train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=keys)
    except ValueError:
        logger.warning("Stratified split failed, falling back to random split.")
        train_idx, val_idx = train_test_split(idx, test_size=val_ratio, random_state=seed)
    return [records[i] for i in train_idx], [records[i] for i in val_idx]


# --------------------------------------------------------------------------- #
# Stage 3: tokenization and instruction masking
# --------------------------------------------------------------------------- #
class MedicalSFTDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        return self.records[i]


class InstructionMaskingCollator:
    """Tokenizes chat-formatted samples and masks everything but the answer.

    The prompt length is measured by rendering the same conversation without
    the assistant turn (``add_generation_prompt=True``); labels for those
    positions and for padding are set to ``IGNORE_INDEX``.
    """

    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.max_length = max_length
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _encode(self, text: str, image: Image.Image | None) -> dict:
        if image is not None:
            out = self.processor(text=text, images=[image], add_special_tokens=False, return_tensors="pt")
        else:
            out = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        return {k: v[0] if k in ("input_ids", "attention_mask") else v for k, v in out.items()}

    def encode_one(self, r: dict) -> dict:
        has_image = r.get("image") is not None
        full_text = self.processor.apply_chat_template(
            build_messages(r["question"], has_image, r["answer"]), tokenize=False
        )
        prompt_text = self.processor.apply_chat_template(
            build_messages(r["question"], has_image), tokenize=False, add_generation_prompt=True
        )
        full = self._encode(full_text, r.get("image"))
        prompt_len = len(self._encode(prompt_text, r.get("image"))["input_ids"])

        input_ids = full["input_ids"]
        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX

        if len(input_ids) > self.max_length:
            # Right truncation; image tokens sit at the start of the user turn
            # so they survive as long as max_length covers the image tiles.
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        item = {"input_ids": input_ids, "labels": labels}
        if "pixel_values" in full:
            item["pixel_values"] = full["pixel_values"]
        return item

    def __call__(self, batch: list[dict]) -> dict:
        items = [self.encode_one(r) for r in batch]
        max_len = max(len(x["input_ids"]) for x in items)
        pad_id = self.tokenizer.pad_token_id

        input_ids = torch.full((len(items), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(items), max_len), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(items), max_len), dtype=torch.long)
        for i, x in enumerate(items):
            n = len(x["input_ids"])
            input_ids[i, :n] = x["input_ids"]          # right padding (Eq. 6)
            labels[i, :n] = x["labels"]
            attention_mask[i, :n] = 1

        out = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        pixel_values = [x["pixel_values"] for x in items if "pixel_values" in x]
        if pixel_values:
            out["pixel_values"] = torch.cat(pixel_values, dim=0)
        return out


def save_records(records: list[dict], path: str) -> None:
    """Write text-only fields of the records to JSONL for inspection."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps({k: v for k, v in r.items() if k != "image"} | {"has_image": r.get("image") is not None}) + "\n")
