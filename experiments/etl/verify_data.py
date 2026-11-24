#!/usr/bin/env python
r"""共享数据层验收自检（只读，不修改任何产物）。

用法：
    python experiments/etl/verify_data.py

检查项对应 PROTOCOL §2 与 lead 2026-11-13 的验收口径：
1. stays 去重后 94,458 行且 stay_id 唯一
2. cohort 472,290 行、两套标签阳性率与 qc.json 一致
3. vitals 非空、六项关键体征的每 stay 覆盖率达标
4. sofa_hourly 覆盖 ≥90% stay 的 hour 0、总分 0–24、均值合理、六项分项齐全
5. 每张表的 parquet 存在、该有 csv 的表有 csv、sha256 与 qc.json 一致
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import polars as pl

WORK = Path(__file__).resolve().parents[2]
OUT = WORK / "experiments" / "data"
CSV_TABLES = {"stays", "cohort", "micro", "transfers", "sofa_hourly"}
TABLES = ["stays", "cohort", "vitals", "labs", "inputevents", "procedureevents",
          "uroutput", "micro", "transfers", "sofa_hourly"]

OK: list[str] = []
BAD: list[str] = []


def check(cond: bool, msg: str) -> None:
    (OK if cond else BAD).append(msg)
    print(("  [OK]   " if cond else "  [FAIL] ") + msg)


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    qc = json.loads((OUT / "qc.json").read_text(encoding="utf-8"))
    print("== 1) stays ==")
    st = pl.read_parquet(OUT / "stays.parquet")
    check(st.height == 94458, f"stays 行数 {st.height} == 94,458")
    check(st["stay_id"].n_unique() == st.height, "stay_id 唯一")
    need = ["intime", "outtime", "los_hours", "first_careunit", "dod", "deathtime",
            "hospital_expire_flag"]
    check(all(c in st.columns for c in need), f"stays 含列 {need}")
    check(bool((st["los_hours"] >= 0).all()), "los_hours 全部 >= 0")

    print("== 2) cohort ==")
    co = pl.read_parquet(OUT / "cohort.parquet")
    check(co.height == 472290, f"cohort 行数 {co.height} == 472,290")
    r1 = float(co["y_death"].mean())
    r2 = float(co["y_dod"].mean())
    check(abs(r1 - qc["cohort"]["rate_y_death"]) < 1e-5,
          f"y_death 阳性率 {r1:.4%} 与 qc 一致")
    check(abs(r2 - qc["cohort"]["rate_y_dod"]) < 1e-5,
          f"y_dod 阳性率 {r2:.4%} 与 qc 一致")
    check(0.010 < r1 < 0.014 and 0.020 < r2 < 0.026, "阳性率落在预期区间（1.19%/2.34%）")
    check(set(co["hour_index"].unique().to_list()) == {0, 1, 2, 4, 6},
          "hour_index 只有 0/1/2/4/6")

    print("== 3) vitals ==")
    vt = pl.read_parquet(OUT / "vitals.parquet", columns=["stay_id", "itemid", "valuenum"])
    check(vt.height > 30_000_000, f"vitals 行数 {vt.height:,} > 3000 万")
    n_stay = st.height
    for iid, name, thr in [(220045, "HR", 0.97), (220210, "RR", 0.97),
                           (220277, "SpO2", 0.97), (220739, "GCS_eye", 0.97),
                           (223901, "GCS_motor", 0.97), (223835, "FiO2", 0.48)]:
        cov = vt.filter(pl.col("itemid") == iid)["stay_id"].n_unique() / n_stay
        check(cov >= thr, f"体征 {name} 每 stay 覆盖率 {cov:.3f} >= {thr}")
    check(int(vt["valuenum"].null_count()) < vt.height * 0.1,
          f"valuenum 非空率 {1 - vt['valuenum'].null_count() / vt.height:.3f} > 0.9")

    print("== 4) sofa_hourly ==")
    so = pl.read_parquet(OUT / "sofa_hourly.parquet")
    h0 = so.filter(pl.col("hour_index") == 0)
    check(h0.height / n_stay >= 0.90, f"hour0 覆盖 {h0.height / n_stay:.3f} >= 0.90")
    check(int(so["sofa_total"].min()) >= 0 and int(so["sofa_total"].max()) <= 24,
          f"sofa_total 范围 [{so['sofa_total'].min()}, {so['sofa_total'].max()}] ⊂ [0,24]")
    m_all = float(so["sofa_total"].mean())
    check(4.0 <= m_all <= 6.0, f"全程 mean(sofa_total)={m_all:.2f} ∈ [4,6]")
    subs = ["sofa_resp", "sofa_coag", "sofa_liver", "sofa_cardio", "sofa_cns", "sofa_renal"]
    check(all(c in so.columns for c in subs), "六项分项分齐全")
    check(bool((h0[subs].sum_horizontal() == h0["sofa_total"]).all()),
          "hour0 六项之和 == sofa_total")
    check(int(h0["hour_index"].max()) == 0 and h0.height > 0, "hour0 切片有效")
    # 信号自检（非实验结论，仅证明体征层非空且方向合理）
    from sklearn.metrics import roc_auc_score
    j = h0.join(co.filter(pl.col("hour_index") == 0), on="stay_id").drop_nulls("y_death")
    auc = roc_auc_score(j["y_death"].to_numpy(), j["sofa_total"].to_numpy())
    check(auc > 0.6, f"信号自检 AUROC(sofa0, y_death)={auc:.3f} > 0.6（体征层确有信号）")

    print("== 5) 文件与校验和 ==")
    for t in TABLES:
        pq = OUT / f"{t}.parquet"
        check(pq.exists(), f"{t}.parquet 存在")
        if pq.exists():
            expect = qc["tables"][t]["parquet"]["sha256"]
            check(sha256(pq) == expect, f"{t}.parquet sha256 与 qc.json 一致")
        cs = OUT / f"{t}.csv"
        if t in CSV_TABLES:
            check(cs.exists(), f"{t}.csv 存在（下游 csv 契约）")
        else:
            check(not cs.exists(), f"{t}.csv 不导出（大表只留 parquet，符合 lead 口径）")
    check((OUT / "qc.json").exists(), "qc.json 存在")

    print("\n" + "=" * 70)
    print(f"通过 {len(OK)} 项，失败 {len(BAD)} 项")
    if BAD:
        print("失败项：")
        for m in BAD:
            print("  - " + m)
        return 1
    print("全部通过：共享数据层可交付。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
