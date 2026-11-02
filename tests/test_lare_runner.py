"""LARE MIMIC 运行器单测：纯合成，不读数据库、不跑 MIMIC、不连网络。"""

import json

import numpy as np
import pytest

from scripts.lare_mimic_run import (
    GATE_ENV,
    GATE_MSG,
    STRATEGIES,
    STRATEGY_METRICS,
    main,
    make_smoke_dataset,
    run_lare_pipeline,
    strategy_masks,
)


def _small(n: int = 400, seed: int = 0):
    return make_smoke_dataset(n=n, seed=seed)


# 1
def test_smoke_dataset_contract():
    p, y_dod, y_death = _small(500, 3)
    assert p.size == y_dod.size == y_death.size == 500
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert set(np.unique(y_death)) <= {0.0, 1.0}
    assert set(np.unique(y_dod)) <= {0.0, 1.0}
    # 粗化标签是精确标签的超集
    assert np.all(y_dod[y_death == 1.0] == 1.0)
    assert y_dod.mean() >= y_death.mean()
    # 同 seed 可复现；不同 seed 不同
    again = make_smoke_dataset(500, 3)
    assert np.array_equal(p, again[0]) and np.array_equal(y_dod, again[1])
    other = make_smoke_dataset(500, 4)
    assert not np.array_equal(p, other[0])
    with pytest.raises(ValueError):
        make_smoke_dataset(5, 0)


# 2
def test_pipeline_summary_keys_complete():
    p, y_dod, y_death = _small()
    payload = run_lare_pipeline(p, y_dod, y_death, [0, 1], alpha=0.3)
    assert payload["config"]["n"] == p.size
    assert payload["config"]["seeds"] == [0, 1]
    assert len(payload["seeds"]) == 2
    assert payload["summary"]["n_seeds"] == 2

    for name in STRATEGIES:
        block = payload["summary"]["strategies"][name]
        for key in STRATEGY_METRICS:
            assert key in block, f"{name} 缺 {key}"
            assert block[key] is None or np.isfinite(block[key])
        assert 0.0 <= (block["budget"] or 0.0) <= 1.0 + 1e-9
        for row in payload["seeds"]:
            per_seed = row["strategies"][name]
            if per_seed["label_free"]:
                # 与标签无关的策略：两套口径下复核集合完全相同
                assert per_seed["reviewed_jaccard"] == pytest.approx(1.0)

    for bound in ("none", "hoeffding"):
        block = payload["summary"]["lar"][bound]
        for key in ("tau_low", "tau_high", "robust_cost", "max_regret", "feasible_rate"):
            assert key in block

    drift = payload["summary"]["label_drift"]
    for key in ("tau_low_dod", "tau_high_dod", "delta_tau_low", "reviewed_jaccard"):
        assert key in drift

    # 必须是严格 JSON（inf/nan 已转 None）
    json.dumps(payload, allow_nan=False)


# 3
def test_summary_aggregates_seed_means():
    p, y_dod, y_death = _small()
    payload = run_lare_pipeline(p, y_dod, y_death, [0, 1], alpha=0.3)
    rows = payload["seeds"]
    for name in STRATEGIES:
        for key in ("review_yield_death", "missed_imminent", "budget"):
            expected = float(np.mean([r["strategies"][name][key] for r in rows]))
            assert payload["summary"]["strategies"][name][key] == pytest.approx(expected)
    assert payload["summary"]["lar"]["none"]["feasible_rate"] == pytest.approx(
        float(np.mean([1.0 if r["lar"]["none"]["feasible"] else 0.0 for r in rows]))
    )
    assert payload["summary"]["base_rate_death"] == pytest.approx(
        float(np.mean([r["base_rate_death"] for r in rows]))
    )


# 4
def test_pipeline_is_deterministic_and_seed_sensitive():
    p, y_dod, y_death = _small()
    a = run_lare_pipeline(p, y_dod, y_death, [0], alpha=0.3)
    b = run_lare_pipeline(p, y_dod, y_death, [0], alpha=0.3)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    c = run_lare_pipeline(p, y_dod, y_death, [1], alpha=0.3)
    assert a["seeds"][0]["base_rate_death"] != c["seeds"][0]["base_rate_death"]

    # 不重采样时，两个不同 seed 的点估计完全一致
    d = run_lare_pipeline(p, y_dod, y_death, [1, 2], alpha=0.3, bootstrap=False)
    assert d["seeds"][0]["strategies"]["fixed"] == d["seeds"][1]["strategies"]["fixed"]


# 5
def test_invalid_inputs_raise_value_error():
    p, y_dod, y_death = _small(200)
    with pytest.raises(ValueError):
        run_lare_pipeline(p, y_dod[:-1], y_death, [0])
    with pytest.raises(ValueError):
        run_lare_pipeline(p, y_dod + 0.5, y_death, [0])
    with pytest.raises(ValueError):
        run_lare_pipeline(p, y_dod, y_death, [])
    with pytest.raises(ValueError):
        run_lare_pipeline(p, y_dod, y_death, [-1])
    with pytest.raises(ValueError):
        run_lare_pipeline(p, y_dod, y_death, [0], K=1.5)
    with pytest.raises(ValueError):
        run_lare_pipeline(p, y_dod, y_death, [0], alpha=0.0)
    with pytest.raises(ValueError):
        run_lare_pipeline(p, y_dod, y_death, [0], delta=1.0)


# 6
def test_strategy_masks_partition_and_double_feasible_path():
    p, _, y_death = _small(400)
    n = p.size
    for name in STRATEGIES:
        masks, meta = strategy_masks(
            name, p, y_death, K=0.2, alpha=0.3, n_grid=50, delta=0.1
        )
        assert set(masks) == {"release", "review", "alert"}
        release, review, alert = (masks[k] for k in ("release", "review", "alert"))
        assert release.size == n
        assert not (release & review).any()
        assert not (release & alert).any()
        assert not (review & alert).any()
        assert np.all(release | review | alert)
        assert isinstance(meta, dict)

    # 宽松 alpha：PAC 双阈值可行，且阈值确实来自 select_double_threshold
    _, meta = strategy_masks("double", p, y_death, K=0.2, alpha=0.5, n_grid=50, delta=0.1)
    assert meta["feasible"] is True
    assert 0.0 <= meta["tau_low"] < meta["tau_high"] <= 1.0

    # deferral 的复核占比 ≈ 预算 K
    masks, _ = strategy_masks("deferral", p, y_death, K=0.2, alpha=0.3, n_grid=50, delta=0.1)
    assert masks["review"].mean() == pytest.approx(0.2, abs=0.03)


# 7
def test_smoke_cli_writes_artifact(tmp_path):
    out = tmp_path / "smoke_lare.json"
    code = main(["--smoke", "--seeds", "1", "--out", str(out)])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["config"]["mode"] == "smoke"
    assert payload["config"]["hours"] == [0, 1, 2, 4, 6]
    assert "summary" in payload and "seeds" in payload
    assert set(payload["summary"]["strategies"]) == set(STRATEGIES)


# 8
def test_real_mode_f1_gate_blocks_without_env(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv(GATE_ENV, raising=False)
    code = main(["--seeds", "1", "--out", str(tmp_path / "should_not_exist.json")])
    assert code == 3
    assert GATE_MSG in capsys.readouterr().out
    assert not (tmp_path / "should_not_exist.json").exists()
