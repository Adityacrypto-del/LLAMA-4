"""Medical benchmark evaluation (Section 3.2.7, Algorithm 3).

Multiple-choice benchmarks use constrained decoding: the next-token logits after
the assistant header are restricted to the valid answer tokens and renormalised
with a softmax, giving calibrated estimates P(a | x). VQA-RAD is evaluated with
greedy generation and text-overlap metrics.

Usage:
    python -m qvaidya.evaluate --config configs/eval_config.yaml
    python -m qvaidya.evaluate --config configs/eval_config.yaml --no-adapter --bf16   # baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm

from .data import (
    MEDQA_LABELS,
    PUBMEDQA_LABELS,
    _take,
    _to_image,
    build_messages,
    medqa_prompt,
    pubmedqa_prompt,
)
from .metrics import classification_metrics, generation_metrics
from .model import load_for_inference, load_processor

logger = logging.getLogger(__name__)


def answer_token_ids(tokenizer, labels: list[str]) -> list[list[int]]:
    """First-token ids for each label and its common surface variants."""
    per_label = []
    for label in labels:
        ids = set()
        for variant in {label, label.lower(), label.capitalize(), " " + label, " " + label.capitalize()}:
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if toks:
                ids.add(toks[0])
        per_label.append(sorted(ids))
    # Variants of different labels must not share a token.
    seen: dict[int, int] = {}
    for i, ids in enumerate(per_label):
        for t in ids:
            if t in seen and seen[t] != i:
                raise ValueError(f"Answer token {t} is shared by labels {labels[seen[t]]!r} and {labels[i]!r}")
            seen[t] = i
    return per_label


@torch.no_grad()
def constrained_probs(model, processor, prompt: str, label_ids: list[list[int]]) -> np.ndarray:
    tokenizer = getattr(processor, "tokenizer", processor)
    text = processor.apply_chat_template(build_messages(prompt, False), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0, -1].float()
    # Restricted softmax over the valid answers (logsumexp over each label's variants).
    scores = torch.stack([torch.logsumexp(logits[ids], dim=0) for ids in label_ids])
    return torch.softmax(scores, dim=0).cpu().numpy()


def eval_multiple_choice(model, processor, samples: list[tuple[str, int]], labels: list[str], n_bins: int) -> dict:
    tokenizer = getattr(processor, "tokenizer", processor)
    label_ids = answer_token_ids(tokenizer, labels)
    probs, gold = [], []
    for prompt, y in tqdm(samples, desc="constrained decoding"):
        probs.append(constrained_probs(model, processor, prompt, label_ids))
        gold.append(y)
    return classification_metrics(np.stack(probs), np.array(gold), n_bins)


def pubmedqa_samples(cfg: dict) -> list[tuple[str, int]]:
    ds = _take(load_dataset(cfg["hf_id"], cfg.get("config", "pqa_labeled"), split=cfg["split"]), cfg.get("num_samples"))
    return [
        (pubmedqa_prompt(s["question"], " ".join(s["context"]["contexts"])), PUBMEDQA_LABELS.index(s["final_decision"].lower()))
        for s in ds
    ]


def medqa_samples(cfg: dict) -> list[tuple[str, int]]:
    ds = _take(load_dataset(cfg["hf_id"], split=cfg["split"]), cfg.get("num_samples"))
    return [(medqa_prompt(s["question"], s["options"]), MEDQA_LABELS.index(s["answer_idx"].upper())) for s in ds]


@torch.no_grad()
def eval_vqa_rad(model, processor, cfg: dict) -> dict:
    ds = _take(load_dataset(cfg["hf_id"], split=cfg["split"]), cfg.get("num_samples"))
    tokenizer = getattr(processor, "tokenizer", processor)
    preds, refs = [], []
    for s in tqdm(ds, desc="VQA-RAD"):
        image = _to_image(s["image"])
        question = f"{s['question']}\nAnswer briefly."
        text = processor.apply_chat_template(build_messages(question, image is not None), tokenize=False, add_generation_prompt=True)
        if image is not None:
            inputs = processor(text=text, images=[image], add_special_tokens=False, return_tensors="pt")
        else:
            inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")
        inputs = inputs.to(model.device)
        out = model.generate(**inputs, max_new_tokens=cfg.get("max_new_tokens", 32), do_sample=False)
        preds.append(tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        refs.append(str(s["answer"]))
    metrics = generation_metrics(preds, refs)
    metrics["examples"] = [{"pred": p, "ref": r} for p, r in list(zip(preds, refs))[:20]]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Q-VAIDYA benchmark evaluation")
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--no-adapter", action="store_true", help="evaluate the base model without LoRA")
    parser.add_argument("--bf16", action="store_true", help="load the backbone in BF16 instead of 8-bit")
    parser.add_argument("--tag", default=None, help="name for the results file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.no_adapter:
        cfg["model"]["adapter_path"] = None
    if args.bf16:
        cfg["model"]["load_in_8bit"] = False

    processor = load_processor(cfg["model"]["name_or_path"])
    model = load_for_inference(cfg["model"])
    bench, n_bins = cfg["benchmarks"], cfg.get("ece_bins", 15)

    results = {}
    if bench.get("pubmedqa", {}).get("enabled"):
        results["pubmedqa"] = eval_multiple_choice(model, processor, pubmedqa_samples(bench["pubmedqa"]), PUBMEDQA_LABELS, n_bins)
    if bench.get("medqa", {}).get("enabled"):
        results["medqa_usmle"] = eval_multiple_choice(model, processor, medqa_samples(bench["medqa"]), MEDQA_LABELS, n_bins)
    if bench.get("vqa_rad", {}).get("enabled"):
        results["vqa_rad_100"] = eval_vqa_rad(model, processor, bench["vqa_rad"])

    tag = args.tag or ("bf16" if not cfg["model"]["load_in_8bit"] else "8bit") + ("_qlora" if cfg["model"].get("adapter_path") else "_base")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    path = os.path.join(cfg["output_dir"], f"results_{tag}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    for name, m in results.items():
        summary = {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}
        logger.info("%s: %s", name, summary)
    logger.info("Results written to %s", path)


if __name__ == "__main__":
    main()
