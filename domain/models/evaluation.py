"""Binary classification metrics for imbalanced mortality prediction."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def net_benefit_at_threshold(
    y_true: Any,
    probability: Any,
    *,
    threshold: float,
) -> dict[str, float]:
    """Decision-curve net benefit at one fixed working-point threshold (H1)."""
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(probability, dtype=float)
    t = float(threshold)
    if not (0.0 < t < 1.0):
        raise ValueError("threshold must be in (0, 1)")
    n = max(len(y), 1)
    pred = (prob >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    prevalence = float(y.mean()) if len(y) else 0.0
    odds = t / max(1.0 - t, 1e-9)
    nb_model = tp / n - fp / n * odds
    nb_all = prevalence - (1.0 - prevalence) * odds
    return {
        "threshold": t,
        "net_benefit_model": float(nb_model),
        "net_benefit_treat_all": float(nb_all),
        "net_benefit_treat_none": 0.0,
        "prevalence": prevalence,
    }


def net_benefit_curve(
    y_true: Any,
    probability: Any,
    thresholds: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Decision-curve analysis style net benefit vs threshold probability."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if thresholds is None:
        thr = np.linspace(0.05, 0.80, 16)
    else:
        thr = np.asarray(list(thresholds), dtype=float)
    nb_model: list[float] = []
    nb_all: list[float] = []
    for t in thr:
        point = net_benefit_at_threshold(y, p, threshold=float(t))
        nb_model.append(point["net_benefit_model"])
        nb_all.append(point["net_benefit_treat_all"])
    return {
        "thresholds": [float(x) for x in thr],
        "net_benefit_model": nb_model,
        "net_benefit_treat_all": nb_all,
        "net_benefit_treat_none": [0.0] * len(thr),
        "n": int(len(y)),
        "prevalence": float(y.mean()) if len(y) else 0.0,
        "status": "ok",
    }


def select_threshold_by_f1(y_true, probability) -> float:
    """Select an operating threshold on validation data only."""
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(probability, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y, prob)
    if len(thresholds) == 0:
        return 0.5
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def binary_metrics(y_true, probability, *, threshold: float = 0.5) -> dict[str, Any]:
    """Return discrimination, calibration and operating-point metrics."""
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(probability, dtype=float)
    if y.ndim != 1 or prob.ndim != 1 or len(y) != len(prob):
        raise ValueError("y_true and probability must be one-dimensional and have equal length")
    if len(y) == 0:
        raise ValueError("cannot evaluate an empty split")
    if np.any((prob < 0) | (prob > 1)):
        raise ValueError("probabilities must be between 0 and 1")

    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    has_both_classes = len(np.unique(y)) > 1
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=10, strategy="quantile")

    nb = (
        net_benefit_at_threshold(y, prob, threshold=float(threshold))
        if 0.0 < float(threshold) < 1.0
        else None
    )
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, prob)) if has_both_classes else None,
        "pr_auc": float(average_precision_score(y, prob)) if has_both_classes else None,
        "brier": float(brier_score_loss(y, prob)),
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "net_benefit_model": None if nb is None else nb["net_benefit_model"],
        "net_benefit_treat_all": None if nb is None else nb["net_benefit_treat_all"],
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "calibration": {
            "mean_predicted": [float(v) for v in mean_pred],
            "fraction_positive": [float(v) for v in frac_pos],
        },
    }
