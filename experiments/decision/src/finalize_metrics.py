"""Produce the final metrics artefacts from the 1000-bootstrap full run and prove that
metrics.json / metrics_final.json agree digit-for-digit with results/tables/table_main.csv.

    python experiments/decision/src/finalize_metrics.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
RES = OUT / "results"

# field name contract used by the report generator
DELONG_KEYS = ("z", "p", "p_two_sided", "log10_p", "auc1", "auc2")


def main(src="metrics_full.json"):
    m = json.loads((RES / src).read_text(encoding="utf-8"))

    # 1. normalise the comparison schema (p_two_sided on every block)
    for key, cmp in m["comparisons"].items():
        d = cmp["delong"]
        d["p_two_sided"] = d["p"]          # DeLong is inherently two-sided
        for nm in ("auprc_delta", "auroc_delta"):
            cmp[nm].setdefault("p_two_sided", None)
        if cmp["auprc_delta"]["p_two_sided"] is None:
            raise SystemExit(f"{key}: missing bootstrap p")
    m["meta"]["n_boot"] = m["comparisons"][next(iter(m["comparisons"]))]["auprc_delta"]["n_boot"]
    m["meta"]["schema"] = {"models": "point metrics on the test fold, scored once",
                           "comparisons": "auprc_delta/auroc_delta = paired patient bootstrap; "
                                          "delong = DeLong AUC test",
                           "delong_keys": list(DELONG_KEYS)}

    # 2. write the deliverable names
    for nm in ("metrics.json", "metrics_final.json"):
        (RES / nm).write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    old = RES / "metrics_vitonly.json"
    if old.exists():
        shutil.move(str(old), str(RES / "metrics_vitonly_prelabs.json"))

    # 3. cross-check against the table the report quotes
    tab = pd.read_csv(RES / "tables" / "table_main.csv")
    fails = []
    for _, row in tab.iterrows():
        name = row["model"]
        if name not in m["models"]:
            fails.append(f"{name}: not in metrics.json")
            continue
        mm = m["models"][name]
        for col, key in (("auprc", "auprc"), ("auroc", "auroc"), ("brier", "brier"),
                         ("ece15", "ece15"), ("cal_slope", "cal_slope")):
            if col in tab.columns and abs(float(row[col]) - float(mm[key])) > 5e-7:
                fails.append(f"{name}.{col}: table={row[col]} metrics={mm[key]}")
        for b in mm["budgets"]:
            suf = f"sens@{b['budget_per_1000']}"
            if suf in tab.columns and b["sensitivity"] is not None:
                if abs(float(row[suf]) - float(b["sensitivity"])) > 5e-7:
                    fails.append(f"{name}.{suf}: table={row[suf]} metrics={b['sensitivity']}")
    print(f"table_main.csv rows: {len(tab)}  models in metrics.json: {len(m['models'])}")
    print(f"n_boot={m['meta']['n_boot']}  comparisons={len(m['comparisons'])}")
    if fails:
        print("MISMATCHES:")
        for f in fails:
            print("  ", f)
        sys.exit(1)
    print("OK: metrics.json == metrics_final.json == table_main.csv (tolerance 5e-7)")
    for key, cmp in m["comparisons"].items():
        print(f"  {key:10s} dAUPRC {cmp['auprc_delta']['abs']:+.4f} "
              f"[{cmp['auprc_delta']['lo']:+.4f},{cmp['auprc_delta']['hi']:+.4f}] "
              f"p_boot={cmp['auprc_delta']['p_two_sided']:.4f} "
              f"DeLong z={cmp['delong']['z']:+.2f} p={cmp['delong']['p_two_sided']:.3e}")


if __name__ == "__main__":
    main()
