"""Evaluation metrics (Section 3.2.7, Algorithm 3)."""

from __future__ import annotations

import re
import string

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


# --------------------------------------------------------------------------- #
# Classification + calibration (PubMedQA, MedQA-USMLE)
# --------------------------------------------------------------------------- #
def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE = sum_m |B_m|/N * |acc(B_m) - conf(B_m)|."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score: 1/N sum_i sum_c (p_ic - y_ic)^2."""
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def classification_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, pred)),
        "ece": expected_calibration_error(probs, labels, n_bins),
        "brier": brier_score(probs, labels),
        "n": int(len(labels)),
    }


# --------------------------------------------------------------------------- #
# Generation quality (Awesome Medical, VQA-RAD-100)
# --------------------------------------------------------------------------- #
_STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "in", "on", "to", "is", "are", "was",
    "were", "be", "with", "for", "by", "as", "at", "this", "that", "it", "from",
}


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return re.sub(r"\s+", " ", text).strip()


def keywords(text: str) -> set[str]:
    return {w for w in normalize(text).split() if w not in _STOPWORDS}


def exact_match(pred: str, ref: str) -> float:
    return float(normalize(pred) == normalize(ref))


def partial_match(pred: str, ref: str) -> float:
    """1 if either normalised string contains the other or they share a keyword."""
    p, r = normalize(pred), normalize(ref)
    if not p or not r:
        return 0.0
    return float(p in r or r in p or bool(keywords(p) & keywords(r)))


def keyword_recall(pred: str, ref: str) -> float:
    ref_kw = keywords(ref)
    if not ref_kw:
        return 1.0
    return len(ref_kw & keywords(pred)) / len(ref_kw)


def generation_metrics(preds: list[str], refs: list[str], bertscore: bool = True) -> dict:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge = {"rouge1": [], "rouge2": [], "rougeL": []}
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        for k in rouge:
            rouge[k].append(s[k].fmeasure)

    out = {k: float(np.mean(v)) if v else 0.0 for k, v in rouge.items()}
    out["exact_match"] = float(np.mean([exact_match(p, r) for p, r in zip(preds, refs)]))
    out["partial_match"] = float(np.mean([partial_match(p, r) for p, r in zip(preds, refs)]))
    out["keyword_recall"] = float(np.mean([keyword_recall(p, r) for p, r in zip(preds, refs)]))
    out["success_rate"] = float(np.mean([bool(p.strip()) for p in preds]))

    if bertscore:
        from bert_score import score as bert_score
        _, _, f1 = bert_score(preds, refs, lang="en", verbose=False)
        out["bertscore_f1"] = float(f1.mean())
    out["n"] = len(preds)
    return out
