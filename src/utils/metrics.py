"""Metrics used throughout the paper's tables.

- macro_f1 is the primary metric (Section "Evaluation Metrics").
- attack_degradation / retention_ratio implement Eq. (attack_degradation) / (retention).
"""
from dataclasses import dataclass, asdict
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


@dataclass
class ClassificationMetrics:
    macro_f1: float
    accuracy: float
    precision: float
    recall: float
    weighted_f1: float
    mcc: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> ClassificationMetrics:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return ClassificationMetrics(
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true.tolist())) > 1 else 0.0,
    )


def attack_degradation(f1_clean: float, f1_attack: float) -> float:
    """Eq. (attack_degradation): Delta = F1_clean - F1_attack."""
    return f1_clean - f1_attack


def retention_ratio(f1_clean: float, f1_attack: float) -> float:
    """Eq. (retention): R = F1_attack / F1_clean. Guards against div-by-zero."""
    if f1_clean == 0:
        return 0.0
    return f1_attack / f1_clean


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}
