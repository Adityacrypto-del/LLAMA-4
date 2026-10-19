import numpy as np
import pytest

from qvaidya.data import IGNORE_INDEX, stratified_split
from qvaidya.metrics import (
    brier_score,
    classification_metrics,
    expected_calibration_error,
    exact_match,
    keyword_recall,
    partial_match,
)


def test_perfect_predictions():
    probs = np.eye(3)[[0, 1, 2, 0]]
    m = classification_metrics(probs, np.array([0, 1, 2, 0]))
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["mcc"] == pytest.approx(1.0)
    assert m["ece"] == pytest.approx(0.0)
    assert m["brier"] == pytest.approx(0.0)


def test_uniform_brier():
    probs = np.full((2, 4), 0.25)
    # (0.75^2 + 3 * 0.25^2) = 0.75
    assert brier_score(probs, np.array([0, 3])) == pytest.approx(0.75)


def test_ece_overconfident():
    probs = np.array([[0.9, 0.1], [0.9, 0.1]])
    # confidence 0.9, accuracy 0.5
    assert expected_calibration_error(probs, np.array([0, 1])) == pytest.approx(0.4)


def test_text_matching():
    assert exact_match("Yes.", "yes") == 1.0
    assert partial_match("left lower lobe pneumonia", "pneumonia") == 1.0
    assert keyword_recall("pleural effusion on the right", "right pleural effusion") == 1.0


def test_stratified_split_keeps_labels():
    records = [{"source": "pubmedqa", "label": lab} for lab in ["yes"] * 60 + ["no"] * 40]
    train, val = stratified_split(records, val_ratio=0.1, seed=0)
    assert len(train) + len(val) == 100
    assert {r["label"] for r in val} == {"yes", "no"}


def test_ignore_index():
    assert IGNORE_INDEX == -100
