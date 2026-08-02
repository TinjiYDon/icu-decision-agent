"""Tune / train secondary page."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
import yaml

from infra.config import load_yaml
from presentation.ui.theme import disclaimer

ROOT = Path(__file__).resolve().parents[2]
FEATURES_YAML = ROOT / "configs" / "features.yaml"
LABELS_YAML = ROOT / "configs" / "labels.yaml"


def render_tune() -> None:
    st.title("调参与重训")
    st.caption("写入建议阈值 / 默认时刻后，使用 --from-existing 重训（不碰 Layer0）")

    feat_cfg = load_yaml("features.yaml")
    lab_cfg = load_yaml("labels.yaml")
    rec = dict(lab_cfg.get("recommend") or {})
    col_a, col_b = st.columns(2)
    with col_a:
        observe = st.number_input("recommend.observe", 0.0, 1.0, float(rec.get("observe", 0.2)), 0.05)
        recheck = st.number_input("recommend.recheck", 0.0, 1.0, float(rec.get("recheck", 0.4)), 0.05)
        monitor = st.number_input("recommend.monitor", 0.0, 1.0, float(rec.get("monitor", 0.7)), 0.05)
    with col_b:
        default_h = int(feat_cfg.get("hour_index", 1))
        hour_index = st.number_input("features.hour_index", 0, 24, default_h, 1)
        st.caption(f"S2 prediction_hours = {feat_cfg.get('prediction_hours')}")

    if st.button("保存配置", type="primary"):
        feat_cfg["hour_index"] = int(hour_index)
        FEATURES_YAML.write_text(
            yaml.safe_dump(feat_cfg, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        lab_cfg["recommend"] = {
            **rec,
            "observe": float(observe),
            "recheck": float(recheck),
            "monitor": float(monitor),
        }
        LABELS_YAML.write_text(
            yaml.safe_dump(lab_cfg, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        st.success("已写入 configs/features.yaml 与 configs/labels.yaml")

    st.divider()
    if st.button("运行训练 --from-existing"):
        with st.spinner("训练中…"):
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT)
            proc = subprocess.run(
                [sys.executable, "-m", "application.train", "--from-existing"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                env=env,
            )
        st.code(proc.stdout or "(no stdout)", language="json")
        if proc.returncode != 0:
            st.error(proc.stderr or f"exit {proc.returncode}")
        else:
            st.success("训练完成 — 请打开「验收」查看指标")
            try:
                st.json(json.loads(proc.stdout))
            except json.JSONDecodeError:
                pass
    disclaimer()
