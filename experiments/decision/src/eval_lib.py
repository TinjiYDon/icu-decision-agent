"""Metric / CI / test library for the decision experiments.

All headline numbers in metrics.json go through this module, and verify.py imports
the same functions, so a recomputation from preds_test.parquet is bit-for-bit
comparable.

Design notes
------------
* Point estimates use sklearn (roc_auc_score / average_precision_score).
* Bootstrap uses a **weighted** implementation over orderings computed once:
  resampling patients = assigning per-row integer multiplicities, so AUC/AP/Brier/
  ECE/sensitivity become O(n) prefix-sum passes instead of 1000 re-sorts.
  The weighted estimators match sklearn exactly on the observed sample (self_test).
* Bootstrap unit = stay_id (patient level): all hours of a resampled stay travel
  together, so the CI reflects between-patient variability, not row count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)


# ---------------------------------------------------------------- basic metrics
def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def ece_score(y, p, n_bins=15):
    """Equal-width expected calibration error over [0, 1]."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    n = len(p)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def calibration_slope_intercept(y, p):
    """Cox calibration: logistic regression y ~ logit(p) -> (intercept, slope)."""
    x = _logit(p).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(x, y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def threshold_at_budget(p, budget_per_1000, w=None):
    p = np.asarray(p, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    n = w.sum()
    k = budget_per_1000 / 1000.0 * n
    order = np.argsort(-p, kind="mergesort")
    cw = np.cumsum(w[order])
    i = int(min(np.searchsorted(cw, k), len(order) - 1))
    return float(p[order][i]), k


def budget_metrics(y, p, budget_per_1000, w=None):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    tau, k = threshold_at_budget(p, budget_per_1000, w)
    alert = p >= tau
    tp = float((y[alert]).sum())
    fp = float(((1 - y)[alert]).sum())
    pos = float(y.sum())
    sens = tp / pos if pos > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    return {
        "budget_per_1000": budget_per_1000,
        "n_alert": int(alert.sum()),
        "alert_rate": float(alert.mean()),
        "threshold": tau,
        "sensitivity": sens,
        "ppv": ppv,
        "nnr": float(1.0 / ppv) if ppv and ppv > 0 else np.nan,
        "tp": int(tp),
        "fp": int(fp),
    }


def dca_curve(y, p, thresholds=None, w=None):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 0.2001, 0.005), 4)
    n = w.sum()
    prev = (y * w).sum() / n
    rows = []
    for t in thresholds:
        a = p >= t
        wa = w[a]
        tp = float((y[a] * wa).sum())
        fp = float(((1 - y)[a] * wa).sum())
        nb = tp / n - fp / n * t / (1 - t)
        nb_all = prev - (1 - prev) * t / (1 - t)
        rows.append({"threshold": float(t), "nb_model": nb, "nb_all": nb_all,
                     "nb_none": 0.0, "net_gain_vs_all": nb - nb_all})
    return pd.DataFrame(rows)


def cost_at_threshold(y, p, tau, cost_miss=4.0, cost_fp=1.0):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    alert = p >= tau
    miss = float(((y == 1) & (~alert)).sum())
    fp = float(((y == 0) & alert).sum())
    n = len(y)
    return {
        "cost_total": (cost_miss * miss + cost_fp * fp) / n,
        "miss": int(miss),
        "fp": int(fp),
        "threshold": float(tau),
    }


def best_cost_threshold_on_val(y, p, cost_miss=4.0, cost_fp=1.0):
    """Single alert threshold minimising 4*miss + 1*fp on VAL only."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    pos = ys.sum()
    n = len(y)
    tp_cum = np.cumsum(ys)
    fp_cum = np.arange(1, n + 1) - tp_cum
    cost = (cost_miss * (pos - tp_cum) + cost_fp * fp_cum) / n
    i = int(np.argmin(cost))
    return float(p[order][i]), int(i + 1), float(cost[i])


def point_metrics(y, p, budgets=(20, 50, 100), n_bins=15):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    out = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "ece15": ece_score(y, p, n_bins),
    }
    icpt, slope = calibration_slope_intercept(y, p)
    out["cal_intercept"] = icpt
    out["cal_slope"] = slope
    out["budgets"] = [budget_metrics(y, p, b) for b in budgets]
    return out


# ---------------------------------------------------------------- fast bundle
class BootBundle:
    """Precomputed orderings + patient-level (weighted) bootstrap.

    Every metric is a function of the per-row multiplicity vector w, and the
    orderings (by score) are computed once, so a replicate is a few O(n) passes.
    """

    def __init__(self, y, p, n_bins=15):
        self.y = np.asarray(y, float)
        self.p = np.asarray(p, float)
        n = len(self.y)
        self.n = n
        o = np.argsort(self.p, kind="mergesort")          # ascending
        self.o_asc = o
        ps, ys = self.p[o], self.y[o]
        self.ps, self.ys = ps, ys
        gid = np.zeros(n, dtype=np.int64)
        if n > 1:
            gid[1:] = np.cumsum(ps[1:] != ps[:-1])
        self.gid = gid
        self.ng = int(gid[-1]) + 1 if n else 0
        first = np.searchsorted(gid, np.arange(self.ng), side="left")
        self.first = first
        # descending order for AP + budget sensitivity
        self.o_desc = o[::-1]
        self.psd, self.ysd = ps[::-1], ys[::-1]
        self.gid_desc = gid[::-1]
        self.gid_rev = (self.gid_desc[0] - self.gid_desc) if n else np.zeros(0, int)
        self.nbin = n_bins
        edges = np.linspace(0, 1, n_bins + 1)
        self.bin = np.clip(np.digitize(self.p, edges[1:-1], right=False), 0, n_bins - 1)

    # -- metrics as functions of w (weights in original row order) -----------
    def _w(self, w):
        return np.asarray(w, float)

    def auc(self, w):
        w = self._w(w)
        wa = w[self.o_asc]
        wp, wn = wa * self.ys, wa * (1 - self.ys)
        Wp, Wn = wp.sum(), wn.sum()
        if Wp <= 0 or Wn <= 0:
            return np.nan
        g_neg = np.bincount(self.gid, weights=wn, minlength=self.ng)
        below = (np.cumsum(wn)[self.first] - wn[self.first])[self.gid]
        return float((wp * (below + 0.5 * g_neg[self.gid])).sum() / (Wp * Wn))

    def ap(self, w):
        w = self._w(w)[self.o_asc][::-1]
        wp, wn = w * self.ysd, w * (1 - self.ysd)
        tp_g = np.bincount(self.gid_rev, weights=wp, minlength=self.ng)
        fp_g = np.bincount(self.gid_rev, weights=wn, minlength=self.ng)
        tp = np.cumsum(tp_g)
        fp = np.cumsum(fp_g)
        Wp = tp[-1]
        if Wp <= 0:
            return np.nan
        prec = tp / np.maximum(tp + fp, 1e-12)
        rec = tp / Wp
        drec = np.diff(np.concatenate([[0.0], rec]))
        return float((drec * prec).sum())

    def brier(self, w):
        w = self._w(w)
        return float((w * (self.p - self.y) ** 2).sum() / w.sum())

    def ece(self, w):
        w = self._w(w)
        W = w.sum()
        wb = np.bincount(self.bin, weights=w, minlength=self.nbin)
        mb = np.bincount(self.bin, weights=w * self.p, minlength=self.nbin)
        yb = np.bincount(self.bin, weights=w * self.y, minlength=self.nbin)
        m = wb > 0
        return float((wb[m] / W * np.abs(mb[m] / wb[m] - yb[m] / wb[m])).sum())

    def sens(self, w, budget_per_1000=50):
        w = self._w(w)[self.o_asc][::-1]
        W = w.sum()
        pos = (self.ysd * w).sum()
        if pos <= 0:
            return np.nan
        cw = np.cumsum(w)
        k = budget_per_1000 / 1000.0 * W
        i = int(min(np.searchsorted(cw, k), len(w) - 1))
        tau = self.psd[i]
        alert = self.psd >= tau
        return float((self.ysd[alert] * w[alert]).sum() / pos)

    def all(self, w):
        return {"auroc": self.auc(w), "auprc": self.ap(w), "brier": self.brier(w),
                "ece15": self.ece(w), "sens@50": self.sens(w, 50)}

    # -- resampling ----------------------------------------------------------
    def _weights_iter(self, stay_ids, n_boot, seed):
        stay_ids = np.asarray(stay_ids)
        uniq, inv = np.unique(stay_ids, return_inverse=True)
        n_stay = len(uniq)
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            draws = rng.integers(0, n_stay, n_stay)
            cnt = np.bincount(draws, minlength=n_stay).astype(float)
            yield cnt[inv]

    def bootstrap(self, stay_ids, n_boot=1000, seed=42, names=None):
        names = names or ("auroc", "auprc", "brier", "ece15", "sens@50")
        acc = {k: np.empty(n_boot) for k in names}
        for b, w in enumerate(self._weights_iter(stay_ids, n_boot, seed)):
            r = self.all(w)
            for k in names:
                acc[k][b] = r[k]
        out = {}
        for k in names:
            v = acc[k][np.isfinite(acc[k])]
            out[k] = {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
                      "hi": float(np.percentile(v, 97.5)), "se": float(v.std(ddof=1)),
                      "n_boot": int(len(v))}
        return out

    def paired_delta(self, other, stay_ids, metric="auprc", n_boot=1000, seed=42):
        """Bootstrap of metric(self) - metric(other) on the same resampled patients."""
        name_map = {"auprc": "ap", "auroc": "auc", "brier": "brier", "ece15": "ece"}
        mname = name_map.get(metric, metric)
        stay_ids = np.asarray(stay_ids)
        uniq, inv = np.unique(stay_ids, return_inverse=True)
        n_stay = len(uniq)
        rng = np.random.default_rng(seed)
        fn_a = getattr(self, mname)
        fn_b = getattr(other, mname)
        d = np.empty(n_boot)
        for b in range(n_boot):
            draws = rng.integers(0, n_stay, n_stay)
            cnt = np.bincount(draws, minlength=n_stay).astype(float)
            w = cnt[inv]
            d[b] = fn_a(w) - fn_b(w)
        d = d[np.isfinite(d)]
        mu = float(np.mean(d))
        pval = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        pval = max(pval, 1.0 / (len(d) + 1))
        return {
            "metric": metric,
            "delta": mu,
            "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)),
            "p_two_sided": float(pval),
            "n_boot": int(len(d)),
        }


def delong_auc(y, p1, p2):
    """DeLong test for two correlated ROC AUCs (row level; ignores patient clustering)."""
    y = np.asarray(y, int)
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        return {"auc1": np.nan, "auc2": np.nan, "z": np.nan, "p": np.nan}

    def midrank(x):
        order = np.argsort(x, kind="mergesort")
        xs = x[order]
        ranks = np.empty(len(x), float)
        i = 0
        n_ = len(xs)
        while i < n_:
            j = i
            while j + 1 < n_ and xs[j + 1] == xs[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
            i = j + 1
        return ranks

    def placement(p):
        tx, ty, tz = midrank(p[pos]), midrank(p[neg]), midrank(p)
        auc = (tz[pos].sum() - m * (m + 1) / 2) / (m * n)
        v10 = (tz[pos] - tx) / n
        v01 = 1.0 - (tz[neg] - ty) / m
        return auc, v10, v01

    a1, v10_1, v01_1 = placement(p1)
    a2, v10_2, v01_2 = placement(p2)
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    cov = s10 / m + s01 / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return {"auc1": float(a1), "auc2": float(a2), "z": 0.0, "p": 1.0}
    z = (a1 - a2) / np.sqrt(var)
    # survival function keeps tiny p-values representable (1-cdf underflows at z>~8)
    p = float(2 * stats.norm.sf(abs(z)))
    return {"auc1": float(a1), "auc2": float(a2), "z": float(z), "p": p,
            "log10_p": float(np.log10(p)) if p > 0 else -323.0}


# ---------------------------------------------------------------- self test
def self_test(seed=0, n=20000):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    p = np.clip(0.05 + 0.35 * y + rng.normal(0, 0.1, n), 0, 1)
    b = BootBundle(y, p)
    w = np.ones(n)
    errs = {
        "auc": abs(b.auc(w) - roc_auc_score(y, p)),
        "ap": abs(b.ap(w) - average_precision_score(y, p)),
        "brier": abs(b.brier(w) - brier_score_loss(y, p)),
        "ece": abs(b.ece(w) - ece_score(y, p)),
        "sens": abs(b.sens(w, 50) - budget_metrics(y, p, 50)["sensitivity"]),
    }
    # also with heavy ties (rounded scores)
    pr = np.round(p, 2)
    br = BootBundle(y, pr)
    errs["auc_ties"] = abs(br.auc(w) - roc_auc_score(y, pr))
    print("weighted-vs-reference abs error:", {k: f"{v:.2e}" for k, v in errs.items()})
    assert max(abs(v) for k, v in errs.items() if not k.endswith("ties")) < 1e-9, errs
    assert errs["auc_ties"] < 1e-9, errs
    return errs


if __name__ == "__main__":
    self_test()
