"""All decision figures (system python + matplotlib, style.py for the shared look).

Run with the SYSTEM python (matplotlib lives there):
    C:\\Users\\lenovo\\AppData\\Local\\Programs\\Python\\Python312\\python.exe \
        experiments\\decision\\src\\figures.py

Writes PNG (300 dpi, CJK) into experiments/decision/figs/ and mirrors them into
experiments/figures/decision/ (the lead's reporting folder).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
RES = OUT / "results"
FIGS = OUT / "figs"
MIRROR = ROOT / "experiments" / "figures" / "decision"
sys.path.insert(0, str(ROOT / "experiments" / "common"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import rcParams  # noqa: E402
import style  # noqa: E402

style.apply()
rcParams["savefig.dpi"] = 300
FIGS.mkdir(parents=True, exist_ok=True)
MIRROR.mkdir(parents=True, exist_ok=True)

LABEL = {"B0": "B0 常数基准率", "B1": "B1 SOFA 总分", "B2": "B2 qSOFA/NEWS2",
         "B3": "B3 逻辑回归（全体征）", "B4": "B4 旧仓 19 特征", "B5": "B5 全体征时序",
         "B6": "B5 + 等渗校准", "B7": "B7 五种子集成"}
COL = {"B0": style.C["gray"], "B1": style.C["gold"], "B2": style.C["orange"],
       "B3": style.C["teal"], "B4": style.C["red"], "B5": style.C["blue"],
       "B6": style.C["purple"], "B7": style.C["ink"]}


def save(fig, name, note=None):
    p = FIGS / name
    fig.savefig(p)
    plt.close(fig)
    shutil.copy2(p, MIRROR / name)
    print(f"[fig] {p.name}" + (f"  ({note})" if note else ""), flush=True)


def load():
    m = json.loads((RES / "metrics_full.json").read_text(encoding="utf-8"))
    p = pd.read_parquet(RES / "preds_test.parquet")
    return m, p


# --------------------------------------------------------------- 1. PR curve
def fig_pr(m, p):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    y = p["y"].values
    fig, ax = style.figure(7.4, 4.6)
    for k in ("B0", "B4", "B3", "B5", "B7"):
        c = f"p_{k}"
        if c not in p.columns:
            continue
        if k == "B0":
            ax.axhline(y.mean(), color=COL[k], ls=":", lw=1.6,
                       label=f"{LABEL[k]}（患病率 {y.mean():.3f}）")
            continue
        pr, rc, _ = precision_recall_curve(y, p[c].values)
        ap = average_precision_score(y, p[c].values)
        ax.plot(rc, pr, color=COL[k], lw=2.2 if k in ("B5", "B7") else 1.7,
                label=f"{LABEL[k]} AP={ap:.3f}")
    ax.set_xlabel("召回率 / Recall（未来 12h 死亡阳性）")
    ax.set_ylabel("精确率 / Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.55, ax.get_ylim()[1]))
    ax.set_title(f"测试集 PR 曲线（未来 12 小时内死亡）：B4 AP={m['models']['B4']['auprc']:.3f}，"
                 f"B5 AP={m['models']['B5']['auprc']:.3f}，B7 AP={m['models']['B7']['auprc']:.3f}")
    ax.legend(loc="upper right")
    save(fig, "fig_pr_curve.png", "AP B4 0.125 vs B5 0.310 vs B7 0.323")


# --------------------------------------------------------------- 2. ROC curve
def fig_roc(m, p):
    from sklearn.metrics import roc_curve, roc_auc_score
    y = p["y"].values
    fig, ax = style.figure(7.0, 4.6)
    for k in ("B4", "B3", "B5", "B7"):
        c = f"p_{k}"
        if c not in p.columns:
            continue
        fpr, tpr, _ = roc_curve(y, p[c].values)
        auc = roc_auc_score(y, p[c].values)
        ax.plot(fpr, tpr, color=COL[k], lw=2.2 if k in ("B5", "B7") else 1.7,
                label=f"{LABEL[k]} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], color=style.C["gray"], ls=":", lw=1.2)
    ax.set_xlabel("假阳性率 / 1 - Specificity")
    ax.set_ylabel("真阳性率 / Sensitivity")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"测试集 ROC 曲线：B4 AUC={m['models']['B4']['auroc']:.3f}，"
                 f"B5 AUC={m['models']['B5']['auroc']:.3f}")
    ax.legend(loc="lower right")
    save(fig, "fig_roc_curve.png", "AUC B4 0.829 vs B5 0.925")


# --------------------------------------------------------------- 3. calibration
def fig_calib(m, p):
    y = p["y"].values
    fig, axes = style.figure(9.6, 4.4, ncols=2)
    ax = axes[0]
    edges = np.linspace(0, 1, 16)
    for k in ("B4", "B5", "B6", "B7"):
        c = f"p_{k}"
        if c not in p.columns:
            continue
        pv = p[c].values
        idx = np.clip(np.digitize(pv, edges[1:-1]), 0, 14)
        xs, ys = [], []
        for b in range(15):
            msk = idx == b
            if msk.sum() < 50:
                continue
            xs.append(pv[msk].mean()); ys.append(y[msk].mean())
        cal = m["models"][k]
        ax.plot(xs, ys, "o-", color=COL[k], ms=4, lw=1.7,
                label=f"{LABEL[k]} slope={cal['cal_slope']:.2f}")
    ax.plot([0, 0.6], [0, 0.6], color=style.C["gray"], ls=":", lw=1.2, label="完美校准")
    ax.set_xlabel("预测风险（15 等宽分箱均值）")
    ax.set_ylabel("实际观察死亡率")
    ax.set_title("可靠性曲线（15 个等宽分箱）")
    ax.legend(loc="upper left", fontsize=8)

    ax2 = axes[1]
    ks = [k for k in ("B4", "B5", "B6", "B7") if f"p_{k}" in p.columns]
    eces = [m["models"][k]["ece15"] * 1000 for k in ks]
    brs = [m["models"][k]["brier"] * 1000 for k in ks]
    x = np.arange(len(ks))
    ax2.bar(x - 0.2, eces, width=0.38, color=style.C["blue"], label="ECE(15) ×1000")
    ax2.bar(x + 0.2, brs, width=0.38, color=style.C["orange"], label="Brier ×1000")
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABEL[k].split(" ", 1)[0] for k in ks])
    ax2.set_ylabel("指标值 ×1000（越小越好）")
    ax2.set_title("Brier 与 ECE（15 分箱）对比")
    for xi, (e, b) in enumerate(zip(eces, brs)):
        ax2.text(xi - 0.2, e, f"{e:.1f}", ha="center", va="bottom", fontsize=7)
        ax2.text(xi + 0.2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=7)
    ax2.legend(loc="upper right")
    save(fig, "fig_calibration.png", "Brier 0.0113→0.0098; B6 slope 0.89")


# --------------------------------------------------------------- 4. alarm budget
def fig_budget(m, p):
    budgets = [20, 50, 100]
    fig, axes = style.figure(9.6, 4.2, ncols=2)
    for ax, key, title, ylab in (
            (axes[0], "sensitivity", "报警预算-检出率", "灵敏度 / Sensitivity（%）"),
            (axes[1], "ppv", "报警预算-阳性预测值", "PPV（%）")):
        for k in ("B4", "B5", "B7"):
            vals, xs = [], []
            for b in m["models"][k]["budgets"]:
                v = b.get(key)
                if v is None:
                    continue
                xs.append(b["budget_per_1000"]); vals.append(v * 100)
            ax.plot(xs, vals, "o-", color=COL[k], lw=2.0, ms=5, label=LABEL[k])
        ax.set_xlabel("报警预算（条 / 每 1000 stay-hour）")
        ax.set_ylabel(ylab)
        ax.set_xticks(budgets)
        ax.set_title(title)
        ax.legend(loc="best")
    s50 = [b["sensitivity"] * 100 for k in ("B4", "B5") for b in m["models"][k]["budgets"]
           if b["budget_per_1000"] == 50]
    axes[0].annotate(f"{s50[0]:.0f}% → {s50[1]:.0f}%\n(+{s50[1]-s50[0]:.0f} pp)",
                     xy=(50, s50[1]), xytext=(62, s50[1] - 14),
                     arrowprops=dict(arrowstyle="->", color=style.C["gray"]), fontsize=9)
    save(fig, "fig_alarm_budget.png", "sens@50/1000 0.392→0.667")


# --------------------------------------------------------------- 5. DCA
def fig_dca(m, p):
    fig, ax = style.figure(7.4, 4.4)
    t = np.array([r["threshold"] for r in m["models"]["B5"]["dca"]])
    for k in ("B4", "B5", "B7"):
        nb = np.array([r["nb_model"] for r in m["models"][k]["dca"]])
        ax.plot(t * 100, nb, color=COL[k], lw=2.0, label=LABEL[k])
    nb_all = np.array([r["nb_all"] for r in m["models"]["B5"]["dca"]])
    ax.plot(t * 100, nb_all, color=style.C["gray"], ls="--", lw=1.3, label="全部报警")
    ax.axhline(0, color=style.C["ink"], lw=1.0, ls=":", label="不报警")
    ax.set_xlabel("决策阈值（预测 12h 死亡率 %）")
    ax.set_ylabel("净收益 / Net benefit")
    ax.set_title("决策曲线：净收益随阈值变化（1%–20%）")
    ax.legend(loc="upper right")
    save(fig, "fig_dca.png", "DCA net benefit B4 vs B5 vs B7")


# --------------------------------------------------------------- 6. importance
def fig_importance(m, p):
    f = RES / "tables" / "table_feature_importance_B5.csv"
    d = pd.read_csv(f).head(20).iloc[::-1]
    fig, ax = style.figure(7.6, 6.0)
    colors = [style.C["orange"] if c.startswith(("v_", "tr_")) else
              style.C["teal"] if c.startswith("l_") else style.C["purple"]
              for c in d["feature"]]
    ax.barh(d["feature"], d["gain"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("LightGBM 分裂增益 / gain")
    ax.set_title("B5 的前 20 个重要特征（LightGBM gain）")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=style.C["orange"], label="体征 v_/趋势 tr_"),
                       Patch(color=style.C["teal"], label="化验 l_"),
                       Patch(color=style.C["purple"], label="其他/评分")],
              loc="lower right")
    ax.tick_params(axis="y", labelsize=8)
    save(fig, "fig_feature_importance.png", "top-20 gain")


# --------------------------------------------------------------- 7. LARE
def fig_lare(m, p):
    lj = json.loads((RES / "lare.json").read_text(encoding="utf-8"))
    reg = pd.DataFrame(lj["regret_test"])
    fig, axes = style.figure(10.0, 4.3, ncols=2)
    ax = axes[0]
    xs = np.arange(len(reg))
    ax.bar(xs - 0.2, reg["cost_death"], width=0.38, color=style.C["red"],
           label="按 y_death（精确）评估")
    ax.bar(xs + 0.2, reg["cost_dod"], width=0.38, color=style.C["orange"],
           label="按 y_dod（日期粗化）评估")
    ax.set_xticks(xs)
    ax.set_xticklabels(["A: 按死亡时间选", "B: 按 dod 选", "R: min-max 鲁棒"])
    ax.set_ylabel("期望代价（漏:误=4:1，复核 0.2）")
    ax.set_title("三种策略在两套死亡标签口径下的期望代价")
    ax.legend(loc="upper left", fontsize=8)

    ax2 = axes[1]
    jac = lj["jaccard_test"]
    names = ["复核名单\nA vs B", "报警名单\nA vs B", "放行名单\nA vs B",
             "复核名单\nA vs R", "复核名单\nB vs R"]
    vals = [jac["review_A_vs_B"], jac["alert_A_vs_B"], jac["release_A_vs_B"],
            jac["review_A_vs_R"], jac["review_B_vs_R"]]
    bars = ax2.bar(np.arange(5), vals, color=[style.C["blue"], style.C["blue_light"],
                                              style.C["gray"], style.C["purple"],
                                              style.C["purple"]])
    ax2.axhline(1.0, color=style.C["ink"], ls=":", lw=1.0)
    ax2.set_xticks(np.arange(5)); ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Jaccard 一致度")
    ax2.set_title(f"名单一致度（复核名单 A vs B Jaccard={jac['review_A_vs_B']:.2f}）")
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    save(fig, "fig_lare_regret.png", f"Jaccard A/B {jac['review_A_vs_B']:.3f}")


# --------------------------------------------------------------- 8. gain waterfall
def fig_waterfall(m):
    ex = json.loads((RES / "extras.json").read_text(encoding="utf-8"))
    ab = {r["variant"]: r for r in ex["ablation"]}
    order = ["add_static", "add_static+labs", "add_static+labs+vitals",
             "add_static+labs+vitals+derived", "full"]
    names = ["静态+合并症", "+化验聚合", "+体征聚合", "+派生评分", "+趋势(=B5)"]
    vals = [ab[k]["auprc"] for k in order if k in ab]
    names = [n for k, n in zip(order, names) if k in ab]
    base = ab["add_static"]["auprc"] if "add_static" in ab else 0.0
    fig, ax = style.figure(8.0, 4.4)
    deltas = [v - base for v in vals]
    cum = [base] + [base + sum(deltas[1:i + 1]) for i in range(1, len(vals))]
    ax.plot(range(len(vals)), vals, "o-", color=style.C["blue"], lw=2.2, ms=6,
            label="累计 AUPRC")
    ax.bar(range(len(vals)), deltas, bottom=base, color=style.C["blue_light"], alpha=0.55,
           width=0.55, label="该组带来的增量")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("AUPRC（test）")
    ax.set_title("特征组的增量贡献（同划分、同超参）")
    ax.legend(loc="upper left")
    save(fig, "fig_gain_waterfall.png", "additive ablation")


def main():
    m, p = load()
    fig_pr(m, p)
    fig_roc(m, p)
    fig_calib(m, p)
    fig_budget(m, p)
    fig_dca(m, p)
    fig_importance(m, p)
    fig_lare(m, p)
    try:
        fig_waterfall(m)
    except Exception as e:  # extras may not be ready yet
        print("skip waterfall:", e)
    print("done")


if __name__ == "__main__":
    main()
