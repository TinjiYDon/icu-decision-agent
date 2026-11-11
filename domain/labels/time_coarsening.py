"""结局时间记录精度：受控粗化标签（D1 的纯函数部分）。

动机：MIMIC 的 `dod` 是日期级记录，`deathtime` 是时间戳；两者不仅有精度差异，
还有来源/覆盖差异（dod 含院外死亡）。主实验应在**同一批有精确时刻**的病例上
人为把事件时间粗化到 1h/6h/12h/24h，再重新判定 12h 窗标签，避免把来源差异
当成时间精度差异。

本模块只做“给定精确事件时间 + 预测时刻 + 分辨率 → 粗化标签”的纯函数：
- 事件时间区间完全落在预测窗 `(t, t+h]` 内 → 1；
- 完全落在窗外 → 0；
- 跨越窗口边界 → `None`（不可确定，不能强行记 0/1）。
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "coarse_interval",
    "coarse_label",
    "coarsen_labels",
]


def _num(value: float, what: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} 必须是数值") from exc
    if not math.isfinite(out):
        raise ValueError(f"{what} 必须是有限数值")
    return out


def coarse_interval(event_time: float, resolution_hours: float) -> tuple[float, float]:
    """事件时间所在的粗化区间 `[start, end)`（按分辨率向下取整）。"""
    t = _num(event_time, "event_time")
    res = _num(resolution_hours, "resolution_hours")
    if res <= 0:
        raise ValueError("resolution_hours 必须 > 0")
    start = math.floor(t / res) * res
    return float(start), float(start + res)


def coarse_label(
    event_time: float,
    prediction_time: float,
    *,
    horizon_hours: float = 12.0,
    resolution_hours: float = 24.0,
) -> float | None:
    """单例粗化标签：1=确定在窗内，0=确定在窗外，None=跨边界不可确定。

    预测窗为 `[prediction_time, prediction_time + horizon_hours)`（与 12h 标签同口径）。
    粗化区间与窗口的重叠关系：
    - 区间完全在窗内 → 1；
    - 区间完全在窗外 → 0；
    - 其余（跨边界）→ None。
    """
    t = _num(prediction_time, "prediction_time")
    h = _num(horizon_hours, "horizon_hours")
    if h <= 0:
        raise ValueError("horizon_hours 必须 > 0")
    start, end = coarse_interval(event_time, resolution_hours)
    window_start = t
    window_end = t + h
    if end <= window_start or start >= window_end:
        return 0.0
    if start >= window_start and end <= window_end:
        return 1.0
    return None


def coarsen_labels(
    event_times: Sequence[float],
    prediction_times: Sequence[float],
    *,
    horizon_hours: float = 12.0,
    resolution_hours: float = 24.0,
) -> dict:
    """批量粗化：返回 `labels`（float，跨边界为 NaN）、`ambiguous` 掩码与统计。

    返回值：
    - `labels`: (n,) float，1/0/NaN；
    - `ambiguous`: (n,) bool，True 表示跨边界；
    - `n_ambiguous`: 跨边界数量；
    - `ambiguity_rate`: 跨边界比例；
    - `resolution_hours` / `horizon_hours`: 口径回显。
    """
    events = np.asarray(list(event_times), dtype=float).reshape(-1)
    preds = np.asarray(list(prediction_times), dtype=float).reshape(-1)
    if events.size == 0:
        raise ValueError("event_times 不能为空")
    if events.size != preds.size:
        raise ValueError(f"长度不一致: {events.size} vs {preds.size}")
    if not np.all(np.isfinite(events)) or not np.all(np.isfinite(preds)):
        raise ValueError("event_times / prediction_times 含 NaN/Inf")
    labels = np.full(events.size, np.nan, dtype=float)
    amb = np.zeros(events.size, dtype=bool)
    for i, (e, p) in enumerate(zip(events, preds)):
        value = coarse_label(
            float(e), float(p),
            horizon_hours=horizon_hours,
            resolution_hours=resolution_hours,
        )
        if value is None:
            amb[i] = True
        else:
            labels[i] = value
    return {
        "labels": labels,
        "ambiguous": amb,
        "n_ambiguous": int(np.count_nonzero(amb)),
        "ambiguity_rate": float(np.mean(amb)),
        "resolution_hours": float(resolution_hours),
        "horizon_hours": float(horizon_hours),
    }
