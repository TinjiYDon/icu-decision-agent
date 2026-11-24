"""统一图表风格（供系统 python + matplotlib 使用）。

用法：
    import sys; sys.path.insert(0, r"<workspace>\experiments\common")
    import style
    style.apply()
    fig, ax = style.figure(...)
    style.save(fig, fig_dir / "fig_d_01_pr_curve.png", note="AUPRC 0.42 vs 0.15")

要求：中文标签用微软雅黑；坐标轴去上右边框；网格浅灰；误差棒 95%CI；200dpi。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# ---- 调色板（报告统一） ----
C = {
    "blue": "#2E5AAC",
    "blue_light": "#8FB0E2",
    "orange": "#E8833A",
    "teal": "#2A9D8F",
    "red": "#C8443F",
    "gold": "#D9A521",
    "gray": "#8A8F98",
    "gray_light": "#D9DCE1",
    "purple": "#6B5B95",
    "ink": "#1B1F24",
}

# 方法名 → 颜色（两个实验通用）
METHOD_COLORS = {
    "FCFS": C["gray"],
    "Greedy-SOFA": C["orange"],
    "Greedy-priority": C["gold"],
    "Greedy+reserve": C["purple"],
    "CP-SAT WS": C["blue_light"],
    "CP-SAT Lex": C["blue"],
    "ε-constraint": C["teal"],
    "Rolling CP-SAT": C["red"],
}

_CJK_CANDIDATES = [
    "Microsoft YaHei",
    "Microsoft JhengHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "SimSun",
]


def _pick_cjk_font() -> str | None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            return name
    return None


def apply(font_size: float = 10.5) -> None:
    """全局 rcParams：一次性设置，可重复调用。"""
    cjk = _pick_cjk_font()
    rcParams.update(
        {
            "font.sans-serif": ([cjk] if cjk else []) + ["DejaVu Sans"],
            "font.family": "sans-serif",
            "axes.unicode_minus": False,
            "font.size": font_size,
            "axes.titlesize": font_size + 1.5,
            "axes.labelsize": font_size,
            "axes.edgecolor": "#3A3F45",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E6E8EB",
            "grid.linewidth": 0.8,
            "xtick.color": "#3A3F45",
            "ytick.color": "#3A3F45",
            "legend.frameon": False,
            "legend.fontsize": font_size - 1,
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
            "lines.linewidth": 1.9,
            "lines.markersize": 5,
        }
    )


def figure(w: float = 7.2, h: float = 4.3, ncols: int = 1, nrows: int = 1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kw)
    return fig, axes


def save(fig, path: str | Path, note: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"[figure] {path}" + (f"  ({note})" if note else ""))
    return path


def ci_bar(ax, xs, means, lo, hi, color, label=None, width=0.6, **kw):
    """竖向条形 + 95%CI 误差棒。"""
    import numpy as np

    means = np.asarray(means, dtype=float)
    err = np.vstack([means - np.asarray(lo, float), np.asarray(hi, float) - means])
    ax.bar(xs, means, width=width, color=color, label=label,
           edgecolor="white", linewidth=0.6, **kw)
    ax.errorbar(xs, means, yerr=err, fmt="none", ecolor="#3A3F45",
                elinewidth=1.1, capsize=3, capthick=1.1)


# ---- 便捷别名（供各实验脚本直接用 style.BLUE 等写法） ----
BLUE = C["blue"]
BLUE_L = C["blue_light"]
ORANGE = C["orange"]
TEAL = C["teal"]
RED = C["red"]
GOLD = C["gold"]
GRAY = C["gray"]
LIGHT = C["gray_light"]
PURPLE = C["purple"]
INK = C["ink"]

# 负荷档（宽松/紧张/危机）等分组用的顺序色
PALETTE = [C["gray_light"], C["blue"], C["orange"], C["teal"], C["purple"]]

