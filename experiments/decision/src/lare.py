"""LARE: label-ambiguity review evaluation + LAR-Policy robust double threshold.

Two death labels for the same row:
  y_death  (icu_decision.paper_label.mortality_12h)      strict, deathtime based
  y_dod    (icu_decision.paper_label.mortality_12h_dod)  coarse, date based, superset

Protocol (PROTOCOL §3 LARE):
 1. On VAL, pick the cost-optimal double threshold (release / review / alert) under
    each label separately, subject to budget K and release-risk alpha.
 2. Cross-evaluate every policy under BOTH labels on TEST: yield, missed mass,
    cost, budget compliance.
 3. LAR-Policy: min-max over the two labels' costs -> robust_cost, max_regret.
 4. Review-list Jaccard between the two label-specific policies.

Cost model (H4/H9): miss 4, false alarm 1, review 0.2 per reviewed row; review
absorbs both miss and false-alarm cost (assumed correctable by a human).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
RES = OUT / "results"
CACHE = OUT / "cache"

K_BUDGET = 0.35
ALPHA = 0.05
COST_FN = 4.0
COST_FP = 1.0
COST_REVIEW = 0.2
N_GRID = 41


def tic(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class Grid:
    """Analytic (tau_low, tau_high) grid over sorted scores."""

    def __init__(self, p, y_death, y_dod, n_grid=N_GRID):
        self.p = np.asarray(p, float)
        self.yd = np.asarray(y_death, float)
        self.yo = np.asarray(y_dod, float)
        self.n = len(p)
        o = np.argsort(self.p, kind="mergesort")
        self.ps = self.p[o]
        self.cum_d = np.concatenate([[0], np.cumsum(self.yd[o])])
        self.cum_o = np.concatenate([[0], np.cumsum(self.yo[o])])
        qs = np.linspace(0, 1, n_grid)
        # candidate thresholds at score quantiles (plus the extremes)
        cand = np.unique(np.quantile(self.ps, qs))
        self.taus = np.round(np.concatenate([[0.0], cand, [1.0]]), 6)
        self.cand = self.taus
        self.n_lt = np.searchsorted(self.ps, self.cand, side="left")   # p < tau
        self.n_le = np.searchsorted(self.ps, self.cand, side="right")
        self.grid = self._build()

    def _build(self):
        rows = []
        T = len(self.cand)
        n = self.n
        for i in range(T):
            for j in range(T):
                if i >= j:
                    continue
                n_lt_low = self.n_lt[i]
                n_lt_high = self.n_lt[j]
                review = n_lt_high - n_lt_low
                alert = n - n_lt_high
                budget = (review + alert) / n
                rows.append((i, j, self.cand[i], self.cand[j], n_lt_low, n_lt_high,
                             review, alert, budget))
        df = pd.DataFrame(rows, columns=["i", "j", "tl", "th", "n_release", "n_lt_high",
                                         "n_review", "n_alert", "budget"])
        # cumulative positives / negatives by label for the analytic cost
        Yd_lt = self.cum_d[self.n_lt]          # y=1 with p < tau (release misses)
        Yo_lt = self.cum_o[self.n_lt]
        Yd_lt_h = self.cum_d[self.n_lt]
        df["rel_d"] = Yd_lt[df["i"].values] / n
        df["rel_o"] = Yo_lt[df["i"].values] / n
        # false alarms: y=0 with p >= tau_high
        tot_d, tot_o = self.yd.sum(), self.yo.sum()
        yd_ge = tot_d - Yd_lt_h[df["j"].values]
        yo_ge = tot_o - Yo_lt[df["j"].values]
        df["fp_d"] = ((self.n - df["n_lt_high"].values) - yd_ge) / n
        df["fp_o"] = ((self.n - df["n_lt_high"].values) - yo_ge) / n
        df["review_frac"] = df["n_review"] / n
        df["cost_d"] = (COST_FN * df["rel_d"] + COST_FP * df["fp_d"]
                        + COST_REVIEW * df["review_frac"])
        df["cost_o"] = (COST_FN * df["rel_o"] + COST_FP * df["fp_o"]
                        + COST_REVIEW * df["review_frac"])
        return df

    def select(self, objective: str, K=K_BUDGET, alpha=ALPHA):
        g = self.grid
        if objective == "death":
            cost, rel = g["cost_d"], g["rel_d"]
        elif objective == "dod":
            cost, rel = g["cost_o"], g["rel_o"]
        else:                                     # robust min-max
            cost = np.maximum(g["cost_d"], g["cost_o"])
            rel = np.maximum(g["rel_d"], g["rel_o"])
        ok = (g["budget"] <= K + 1e-12) & (rel <= alpha + 1e-12)
        sub = g[ok]
        if sub.empty:
            return {"feasible": False, "tl": 0.0, "th": 1.0, "robust_cost": float("inf"),
                    "cost_d": float("inf"), "cost_o": float("inf"),
                    "budget": 1.0, "n_release": 0}
        c = cost[ok]
        best = np.min(c)
        tie = sub[np.abs(c - best) <= 1e-12]
        # prefer more releases, then smaller tau_high (wider review band)
        tie = tie.sort_values(["n_release", "th"], ascending=[False, True])
        r = tie.iloc[0]
        return {"feasible": True, "tl": float(r["tl"]), "th": float(r["th"]),
                "cost_d": float(r["cost_d"]), "cost_o": float(r["cost_o"]),
                "robust_cost": float(max(r["cost_d"], r["cost_o"])),
                "budget": float(r["budget"]), "n_release": int(r["n_release"])}


def masks(p, tl, th):
    p = np.asarray(p, float)
    release = p < tl
    alert = p >= th
    review = ~release & ~alert
    return release, review, alert


def eval_policy(p, y_death, y_dod, tl, th, K=K_BUDGET, alpha=ALPHA):
    release, review, alert = masks(p, tl, th)
    n = len(p)
    yd = np.asarray(y_death, float)
    yo = np.asarray(y_dod, float)
    out = {"tl": float(tl), "th": float(th),
           "release_frac": float(release.mean()), "review_frac": float(review.mean()),
           "alert_frac": float(alert.mean()),
           "budget": float((review | alert).mean())}
    for name, y in (("dod", yo), ("death", yd)):
        rev = int(review.sum())
        out[f"yield_{name}"] = float(y[review].mean()) if rev else float("nan")
        out[f"missed_mass_{name}"] = float((y * release).mean())
        cost = (COST_FN * (y * release).mean()
                + COST_FP * ((1 - y) * alert).mean()
                + COST_REVIEW * review.mean())
        out[f"cost_{name}"] = float(cost)
        out[f"fn_{name}"] = int(((y == 1) & release).sum())
        out[f"fp_{name}"] = int(((y == 0) & alert).sum())
    out["compliance"] = bool(out["budget"] <= K + 1e-12
                             and max(out["missed_mass_dod"], out["missed_mass_death"]) <= alpha + 1e-12)
    return out


def jaccard(a, b):
    inter = int((a & b).sum())
    uni = int((a | b).sum())
    return 1.0 if uni == 0 else inter / uni


def eval_pairs(pte, yte_d, yte_o, tls, ths):
    """Analytic cost of each (tl, th) pair on a score vector (same cost model)."""
    pte = np.asarray(pte, float)
    yd = np.asarray(yte_d, float)
    yo = np.asarray(yte_o, float)
    o = np.argsort(pte, kind="mergesort")
    ps = pte[o]
    cd = np.concatenate([[0], np.cumsum(yd[o])])
    co = np.concatenate([[0], np.cumsum(yo[o])])
    n = len(pte)
    i_lt = np.searchsorted(ps, np.asarray(tls, float), side="left")
    j_lt = np.searchsorted(ps, np.asarray(ths, float), side="left")
    rev = j_lt - i_lt
    alert = n - j_lt
    rel_d, rel_o = cd[i_lt] / n, co[i_lt] / n
    fp_d = (alert - (yd.sum() - cd[j_lt])) / n
    fp_o = (alert - (yo.sum() - co[j_lt])) / n
    rf = rev / n
    return (COST_FN * rel_d + COST_FP * fp_d + COST_REVIEW * rf,
            COST_FN * rel_o + COST_FP * fp_o + COST_REVIEW * rf,
            (rev + alert) / n)


def regret_distribution(g, pte, yte_d, yte_o, K=K_BUDGET, alpha=ALPHA):
    """Max-regret for every policy that is (budget, release-risk) feasible on VAL,
    evaluated on TEST. Descriptive diagnostic only -- never used for selection."""
    gd = g.grid
    feas = (gd["budget"].values <= K + 1e-12) & \
           (np.maximum(gd["rel_d"].values, gd["rel_o"].values) <= alpha + 1e-12)
    sub = gd[feas]
    if sub.empty:
        return pd.DataFrame()
    cd, co, budget = eval_pairs(pte, yte_d, yte_o, sub["tl"].values, sub["th"].values)
    best_d, best_o = cd.min(), co.min()
    out = pd.DataFrame({
        "tl": sub["tl"].values, "th": sub["th"].values, "budget": budget,
        "cost_death": cd, "cost_dod": co,
        "regret_death": cd - best_d, "regret_dod": co - best_o,
    })
    out["max_regret"] = out[["regret_death", "regret_dod"]].max(axis=1)
    out["robust_cost"] = out[["cost_death", "cost_dod"]].max(axis=1)
    return out.sort_values("max_regret")


def run(pval, yva_d, yva_o, pte, yte_d, yte_o, tag="B5", K=K_BUDGET, alpha=ALPHA):
    t0 = time.time()
    g = Grid(pval, yva_d, yva_o)
    pol = {
        "A_death": g.select("death", K, alpha),
        "B_dod": g.select("dod", K, alpha),
        "R_robust": g.select("robust", K, alpha),
    }
    tic(f"LARE grid {len(g.grid)} pairs; selected: "
        + ", ".join(f"{k}({v['tl']:.3f},{v['th']:.3f})" for k, v in pol.items()))

    # cross evaluation on TEST (policy chosen on val, evaluated on held-out test)
    rows = []
    for pname, pc in pol.items():
        ev = eval_policy(pte, yte_d, yte_o, pc["tl"], pc["th"], K, alpha)
        rows.append({"policy": pname, "chosen_on": pname.split("_")[-1],
                     "val_tl": pc["tl"], "val_th": pc["th"],
                     "val_budget": pc["budget"], "val_robust_cost": pc["robust_cost"], **ev})
    tab = pd.DataFrame(rows)

    # regret matrix on test costs
    best = {"death": tab["cost_death"].min(), "dod": tab["cost_dod"].min()}
    regret = pd.DataFrame({
        "policy": tab["policy"],
        "cost_death": tab["cost_death"], "cost_dod": tab["cost_dod"],
        "regret_death": tab["cost_death"] - best["death"],
        "regret_dod": tab["cost_dod"] - best["dod"],
    })
    regret["max_regret"] = regret[["regret_death", "regret_dod"]].max(axis=1)
    regret["robust_cost"] = regret[["cost_death", "cost_dod"]].max(axis=1)

    # review-list agreement between the two label-specific policies (on test)
    relA, revA, _ = masks(pte, pol["A_death"]["tl"], pol["A_death"]["th"])
    relB, revB, _ = masks(pte, pol["B_dod"]["tl"], pol["B_dod"]["th"])
    relR, revR, _ = masks(pte, pol["R_robust"]["tl"], pol["R_robust"]["th"])
    jac = {
        "review_A_vs_B": jaccard(revA, revB),
        "alert_A_vs_B": jaccard(pte >= pol["A_death"]["th"], pte >= pol["B_dod"]["th"]),
        "review_A_vs_R": jaccard(revA, revR),
        "review_B_vs_R": jaccard(revB, revR),
        "release_A_vs_B": jaccard(relA, relB),
        "n_review_A": int(revA.sum()), "n_review_B": int(revB.sum()), "n_review_R": int(revR.sum()),
    }

    # label-only disagreement view: which rows differ
    lab = {
        "n_test": int(len(yte_d)),
        "pos_death": int(np.sum(yte_d)), "pos_dod": int(np.sum(yte_o)),
        "dod_only": int(np.sum((np.asarray(yte_o) == 1) & (np.asarray(yte_d) == 0))),
    }

    res = {
        "config": {"K_budget": K, "alpha": alpha, "cost_fn": COST_FN, "cost_fp": COST_FP,
                   "review_cost": COST_REVIEW, "n_grid": N_GRID, "score_model": tag,
                   "selection": "val", "evaluation": "test"},
        "policies_val": pol,
        "table_test": tab.to_dict(orient="records"),
        "regret_test": regret.to_dict(orient="records"),
        "jaccard_test": jac,
        "labels_test": lab,
        "repo_reference": {"jaccard_mimic_old": 0.2429, "robust_cost_mimic_old": 0.0708,
                           "max_regret_mimic_old": 0.0039},
        "elapsed_s": round(time.time() - t0, 1),
    }
    # headline numbers
    rrow = regret[regret["policy"] == "R_robust"].iloc[0]
    dist = regret_distribution(g, pte, yte_d, yte_o, K, alpha)
    res["headline"] = {
        "robust_cost": float(rrow["robust_cost"]),
        "max_regret": float(rrow["max_regret"]),
        "jaccard_review_A_vs_B": float(jac["review_A_vs_B"]),
        "yield_gap_dod_minus_death_on_B": float(
            tab[tab["policy"] == "B_dod"]["yield_dod"].iloc[0]
            - tab[tab["policy"] == "B_dod"]["yield_death"].iloc[0]),
        "budget_all_compliant": bool(tab["compliance"].all()),
        "regret_dist_n_policies": int(len(dist)),
        "regret_dist_median": float(dist["max_regret"].median()) if len(dist) else None,
        "regret_dist_p90": float(dist["max_regret"].quantile(0.9)) if len(dist) else None,
    }
    return res, tab, regret, dist


def sensitivity(pval, yva_d, yva_o, pte, yte_d, yte_o, tag="B5"):
    """How robust_cost / max_regret / Jaccard move with the budget K and alpha."""
    rows = []
    for K in (0.05, 0.10, 0.20, 0.35):
        for alpha in (0.001, 0.003, 0.008, 0.05):
            g = Grid(pval, yva_d, yva_o)
            pol = {"A_death": g.select("death", K, alpha),
                   "B_dod": g.select("dod", K, alpha),
                   "R_robust": g.select("robust", K, alpha)}
            if not pol["R_robust"]["feasible"]:
                rows.append({"K": K, "alpha": alpha, "feasible": False})
                continue
            relA, revA, alA = masks(pte, pol["A_death"]["tl"], pol["A_death"]["th"])
            relB, revB, alB = masks(pte, pol["B_dod"]["tl"], pol["B_dod"]["th"])
            relR, revR, alR = masks(pte, pol["R_robust"]["tl"], pol["R_robust"]["th"])
            r = eval_policy(pte, yte_d, yte_o, pol["R_robust"]["tl"], pol["R_robust"]["th"], K, alpha)
            a = eval_policy(pte, yte_d, yte_o, pol["A_death"]["tl"], pol["A_death"]["th"], K, alpha)
            b = eval_policy(pte, yte_d, yte_o, pol["B_dod"]["tl"], pol["B_dod"]["th"], K, alpha)
            note = (revR & (relA | alA)).sum() + ((relA | alA) & ~(relR | alR)).sum()
            rows.append({
                "K": K, "alpha": alpha, "feasible": True,
                "robust_tl": pol["R_robust"]["tl"], "robust_th": pol["R_robust"]["th"],
                "robust_cost": r["cost_d"] if False else max(r["cost_death"], r["cost_dod"]),
                "max_regret": max(r["cost_death"] - min(a["cost_death"], b["cost_death"]),
                                  r["cost_dod"] - min(a["cost_dod"], b["cost_dod"])),
                "jaccard_review_AB": jaccard(revA, revB),
                "review_frac_robust": r["review_frac"],
                "alert_frac_robust": r["alert_frac"],
                "release_frac_robust": r["release_frac"],
                "budget_compliant": r["compliance"],
                "yield_death_robust": r["yield_death"],
                "yield_dod_robust": r["yield_dod"],
            })
    return pd.DataFrame(rows)


def main(scores="preds_full"):
    tag = scores.split("_", 1)[-1]
    pv = pd.read_parquet(CACHE / f"preds_val_{tag}.parquet")
    pt = pd.read_parquet(CACHE / f"preds_test_{tag}.parquet")
    col = "p_B5"
    res, tab, regret, dist = run(pv[col].values, pv["y"].values, pv["y_dod"].values,
                                 pt[col].values, pt["y"].values, pt["y_dod"].values, tag=col[2:])
    sens = sensitivity(pv[col].values, pv["y"].values, pv["y_dod"].values,
                       pt[col].values, pt["y"].values, pt["y_dod"].values, tag=col[2:])
    res["sensitivity"] = sens.to_dict(orient="records")
    (RES / "tables").mkdir(parents=True, exist_ok=True)
    tab.to_csv(RES / "tables" / "table_lare_policies.csv", index=False)
    regret.to_csv(RES / "tables" / "table_lare_regret.csv", index=False)
    dist.to_csv(RES / "tables" / "table_lare_regret_dist.csv", index=False)
    sens.to_csv(RES / "tables" / "table_lare_sensitivity.csv", index=False)
    (RES / "lare.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res["headline"], ensure_ascii=False, indent=1))
    print(tab[["policy", "tl", "th", "yield_death", "yield_dod", "cost_death", "cost_dod",
               "budget"]].to_string(index=False))
    print(sens[["K", "alpha", "robust_cost", "max_regret", "jaccard_review_AB",
                "budget_compliant"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "preds_full")
