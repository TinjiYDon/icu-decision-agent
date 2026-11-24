"""Refresh the DeLong block of an existing metrics.json from preds_test.parquet.

Needed because the first full run stored p-values computed as 2*(1-cdf), which
underflows to 0 for |z| > ~8. The recomputation uses the same eval_lib.delong_auc
(survival-function form) on the delivered predictions, so verify.py reproduces it.

    python experiments/decision/src/refresh_delong.py [tag]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
RES = OUT / "results"
sys.path.insert(0, str(HERE))
import eval_lib as E  # noqa: E402


def main(tag="full"):
    mpath = RES / f"metrics_{tag}.json"
    m = json.loads(mpath.read_text(encoding="utf-8"))
    p = pd.read_parquet(RES / "preds_test.parquet")
    y = p["y"].values.astype(int)
    n_fixed = 0
    for key, cmp in m["comparisons"].items():
        a, b = key.split("_vs_")
        d = E.delong_auc(y, p[f"p_{a}"].values, p[f"p_{b}"].values)
        cmp["delong"] = d
        n_fixed += 1
    mpath.write_text(json.dumps(m, ensure_ascii=False, indent=1))
    print(f"refreshed {n_fixed} DeLong entries in {mpath.name}")
    for key, cmp in m["comparisons"].items():
        d = cmp["delong"]
        print(f"  {key:10s} AUC1={d['auc1']:.4f} AUC2={d['auc2']:.4f} "
              f"z={d['z']:.2f} p={d['p']:.3e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
