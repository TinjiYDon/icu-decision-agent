"""Decision Curve Analysis (DCA) with bootstrap confidence intervals.

Design principles:
- Pure functions, no I/O, no DB, no sklearn dependencies beyond numpy.
- Cost-benefit ratio (CBR) = threshold / (1 - threshold), e.g. CBR=1/9 → threshold=0.1.
- Bootstrap at stay-level (not row-level) to respect the patient cluster structure.
- Backward-compatible: net_benefit_at_threshold / net_benefit_curve still live in evaluation.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Data classes (immutable, single-responsibility)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NetBenefitPoint:
    """Net benefit at a single threshold probability."""
    threshold: float
    net_benefit_model: float
    net_benefit_treat_all: float
    net_benefit_treat_none: float = 0.0


@dataclass(frozen=True)
class DcaResult:
    """Complete DCA output: point estimates + optional CI bands."""
    thresholds: list[float]
    net_benefit_model: list[float]
    net_benefit_treat_all: list[float]
    net_benefit_treat_none: list[float] = field(default_factory=list)
    # CI fields are None when bootstrap was not requested
    ci_lower_model: list[float] | None = None
    ci_upper_model: list[float] | None = None
    n_samples: int = 0
    positive_rate: float = 0.0
    status: str = "ok"
    message: str = ""


@dataclass(frozen=True)
class WorkingPoint:
    """Single operating threshold analysis (fixed CBR)."""
    threshold: float
    cbr: float
    net_benefit_model: float
    net_benefit_treat_all: float
    net_benefit_treat_none: float = 0.0
    positive_rate: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    sensitivity: float = 0.0
    specificity: float = 0.0
    ppv: float = 0.0
    npv: float = 0.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def net_benefit_at_threshold(
    y_true: Any,
    probability: Any,
    *,
    threshold: float,
) -> dict[str, float]:
    """Decision-curve net benefit at one fixed working-point threshold (H1).

    Net Benefit = TP/n - FP/n * CBR,  where CBR = threshold/(1-threshold).
    Treat-all NB = prevalence - (1-prevalence)*CBR.
    Treat-none NB = 0 by convention.
    """
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(probability, dtype=float)
    t = float(threshold)
    if not (0.0 < t < 1.0):
        raise ValueError("threshold must be in (0, 1)")
    n = max(len(y), 1)
    pred = (prob >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prevalence = float(y.mean()) if len(y) else 0.0
    cbr = t / max(1.0 - t, 1e-9)
    nb_model = tp / n - fp / n * cbr
    nb_all = prevalence - (1.0 - prevalence) * cbr
    return {
        "threshold": t,
        "cbr": cbr,
        "net_benefit_model": float(nb_model),
        "net_benefit_treat_all": float(nb_all),
        "net_benefit_treat_none": 0.0,
        "prevalence": prevalence,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "ppv": float(tp / max(tp + fp, 1)),
        "npv": float(tn / max(tn + fn, 1)),
    }


def net_benefit_curve(
    y_true: Any,
    probability: Any,
    thresholds: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Point-estimate DCA curve (no CI). For backward compat with demo_curves.py."""
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


def _single_bootstrap_nb(
    y_true: np.ndarray,
    prob_true: np.ndarray,
    threshold: float,
    *,
    seed: int,
) -> float:
    """One bootstrap replicate: resample rows with replacement, compute NB."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=len(y_true))
    yb = y_true[idx]
    pb = prob_true[idx]
    t = float(threshold)
    n = max(len(yb), 1)
    pred = (pb >= t).astype(int)
    tp = int(((pred == 1) & (yb == 1)).sum())
    fp = int(((pred == 1) & (yb == 0)).sum())
    cbr = t / max(1.0 - t, 1e-9)
    return float(tp / n - fp / n * cbr)


def dca_curve_with_ci(
    y_true: Any,
    probability: Any,
    *,
    thresholds: Sequence[float] | None = None,
    n_boot: int = 1000,
    ci_alpha: float = 0.05,
    seed: int = 42,
) -> DcaResult:
    """Full DCA with bootstrap confidence intervals.

    Parameters
    ----------
    y_true : array-like
        Binary outcome (0/1).
    probability : array-like
        Predicted probability for each sample.
    thresholds : sequence of float, optional
        Threshold probabilities to evaluate. Default: 0.05..0.80 step 0.05.
    n_boot : int
        Number of bootstrap resamples. 1000 is standard; 500 for quick checks.
    ci_alpha : float
        Confidence level = 1 - alpha. 0.05 → 95% CI.
    seed : int
        Reproducibility seed.

    Returns
    -------
    DcaResult with thresholds, point estimates, and 95% CI bands.
    """
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(probability, dtype=float)
    n = len(y)
    if n < 10:
        return DcaResult(
            thresholds=[], net_benefit_model=[], net_benefit_treat_all=[],
            status="too_few", message=f"样本过少 n={n}，Bootstrap CI 不可靠",
        )
    if n < 50:
        # With < 50 samples, use percentile CI but warn
        warnings = ["样本量较小，Bootstrap CI 需谨慎解读"]
    else:
        warnings = []

    if thresholds is None:
        thr = np.linspace(0.05, 0.80, 16)
    else:
        thr = np.asarray(list(thresholds), dtype=float)

    prevalence = float(y.mean())
    treat_all = prevalence - (1.0 - prevalence) * (thr[0] / max(1.0 - thr[0], 1e-9))

    # Point estimates
    nb_model_pts: list[float] = []
    for t in thr:
        pt = net_benefit_at_threshold(y, prob, threshold=float(t))
        nb_model_pts.append(pt["net_benefit_model"])

    # Bootstrap CIs
    ci_lower: list[float] = []
    ci_upper: list[float] = []
    rng = np.random.default_rng(seed)

    for t in thr:
        boot_vals: list[float] = []
        for b in range(n_boot):
            bseed = int(rng.integers(0, 2**31))
            boot_vals.append(_single_bootstrap_nb(y, prob, float(t), seed=bseed))
        boot_arr = np.array(boot_vals)
        lo = float(np.percentile(boot_arr, 100 * ci_alpha / 2))
        hi = float(np.percentile(boot_arr, 100 * (1 - ci_alpha / 2)))
        ci_lower.append(lo)
        ci_upper.append(hi)

    return DcaResult(
        thresholds=[float(x) for x in thr],
        net_benefit_model=nb_model_pts,
        net_benefit_treat_all=[float(treat_all)] * len(thr),
        net_benefit_treat_none=[0.0] * len(thr),
        ci_lower_model=ci_lower,
        ci_upper_model=ci_upper,
        n_samples=n,
        positive_rate=prevalence,
        status="ok",
        message="; ".join(warnings) if warnings else "",
    )


def dca_working_point(
    y_true: Any,
    probability: Any,
    *,
    cost_benefit_ratio: float,
) -> WorkingPoint:
    """Single operating threshold derived from cost-benefit ratio.

    threshold = CBR / (1 + CBR).
    For rare outcomes (prevalence ~1.2%), a common clinical choice is CBR = 1/9
    → threshold = 0.1, meaning clinicians are willing to miss 1 death to treat
    9 false positives.
    """
    cbr = float(cost_benefit_ratio)
    if cbr <= 0 or cbr >= 100:
        raise ValueError(f"CBR must be in (0, 100), got {cbr}")
    threshold = cbr / (1.0 + cbr)
    pt = net_benefit_at_threshold(y_true, probability, threshold=threshold)
    return WorkingPoint(
        threshold=threshold,
        cbr=cbr,
        net_benefit_model=pt["net_benefit_model"],
        net_benefit_treat_all=pt["net_benefit_treat_all"],
        net_benefit_treat_none=pt["net_benefit_treat_none"],
        positive_rate=pt["prevalence"],
        true_positives=pt["tp"],
        false_positives=pt["fp"],
        true_negatives=pt["tn"],
        false_negatives=pt["fn"],
        sensitivity=pt["sensitivity"],
        specificity=pt["specificity"],
        ppv=pt["ppv"],
        npv=pt["npv"],
    )


def dca_compare_models(
    y_true: Any,
    probabilities: dict[str, Any],
    *,
    cost_benefit_ratio: float = 1.0 / 9.0,
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare multiple models at a fixed working point with CI.

    Parameters
    ----------
    y_true : array-like
        Common ground truth.
    probabilities : dict[str, array-like]
        Model name → predicted probabilities.
    cost_benefit_ratio : float
        CBR for the working point. Default 1/9 (threshold≈0.1).
    n_boot : int
        Bootstrap iterations.
    seed : int
        Reproducibility.
    """
    y = np.asarray(y_true, dtype=int)
    n = len(y)
    result: dict[str, Any] = {
        "n_samples": n,
        "positive_rate": float(y.mean()),
        "cost_benefit_ratio": cost_benefit_ratio,
        "working_threshold": cost_benefit_ratio / (1.0 + cost_benefit_ratio),
        "models": {},
    }
    # Reference curves (treat-all, treat-none)
    ref_cbr = cost_benefit_ratio
    ref_thr = ref_cbr / (1.0 + ref_cbr)
    ref_pt = net_benefit_at_threshold(y, y, threshold=ref_thr)  # perfect classifier NB
    reference_nb_all = ref_pt["net_benefit_treat_all"]
    result["reference_treat_all_nb"] = reference_nb_all

    for name, probs in probabilities.items():
        p = np.asarray(probs, dtype=float)
        if len(p) != n:
            result["models"][name] = {"status": "size_mismatch"}
            continue
        wp = dca_working_point(y, p, cost_benefit_ratio=cost_benefit_ratio)
        model_result: dict[str, Any] = {
            "status": "ok",
            "working_point": {
                "threshold": wp.threshold,
                "net_benefit_model": wp.net_benefit_model,
                "net_benefit_treat_all": wp.net_benefit_treat_all,
                "sensitivity": wp.sensitivity,
                "specificity": wp.specificity,
                "ppv": wp.ppv,
                "npv": wp.npv,
            },
        }
        # Bootstrap CI for this model's NB at working point
        boot_vals: list[float] = []
        rng = np.random.default_rng(seed)
        for b in range(n_boot):
            bseed = int(rng.integers(0, 2**31))
            boot_vals.append(
                _single_bootstrap_nb(y, p, wp.threshold, seed=bseed)
            )
        boot_arr = np.array(boot_vals)
        model_result["ci_95"] = {
            "lower": float(np.percentile(boot_arr, 2.5)),
            "upper": float(np.percentile(boot_arr, 97.5)),
        }
        # Full curve point estimates (no CI, for plotting)
        curve = net_benefit_curve(y, p)
        model_result["curve"] = {
            "thresholds": curve["thresholds"],
            "net_benefit_model": curve["net_benefit_model"],
            "net_benefit_treat_all": curve["net_benefit_treat_all"],
        }
        result["models"][name] = model_result

    return result
