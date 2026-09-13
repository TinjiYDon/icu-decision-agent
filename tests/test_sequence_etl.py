"""Sparse gate for GRU-D sequence ETL."""

from __future__ import annotations

import numpy as np

from domain.features.sequence_etl import passes_sparse_gate


def test_passes_sparse_gate_rejects_empty():
    m = np.zeros((12, 8), dtype=np.float64)
    assert passes_sparse_gate(m) is False


def test_passes_sparse_gate_accepts_vitals():
    m = np.zeros((12, 8), dtype=np.float64)
    # FEATURE_NAMES: hr,sbp,lactate,creatinine,resp_rate,temperature,spo2,bun
    m[:, 0] = 1.0
    m[:4, 1] = 1.0
    assert passes_sparse_gate(m, feature_names_list=[
        "hr", "sbp", "lactate", "creatinine", "resp_rate", "temperature", "spo2", "bun"
    ]) is True
