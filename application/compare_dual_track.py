"""Compare LGBM vs GRU-D metrics JSON (dual-track答辩对照)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "models"

KEYS = (
    "roc_auc",
    "pr_auc",
    "brier",
    "n",
    "test_n",
    "positive",
    "pos_rate",
    "operating_threshold",
    "net_benefit_test_at_operating",
    "net_benefit_treat_all_at_operating",
)


def _load(path: Path) -> dict:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = "ok"
    data["path"] = str(path)
    return data


def compare() -> dict:
    lgbm = _load(ART / "metrics_mortality_12h.json")
    grud = _load(ART / "metrics_grud.json")
    row = {"lgbm": {}, "grud": {}, "notes": []}
    for k in KEYS:
        if lgbm.get("status") == "ok" and k in lgbm:
            row["lgbm"][k] = lgbm[k]
        if grud.get("status") == "ok" and k in grud:
            row["grud"][k] = grud[k]
    if lgbm.get("status") != "ok":
        row["notes"].append("LGBM metrics missing — run application.train")
    if grud.get("status") != "ok":
        row["notes"].append("GRU-D metrics missing — run application.train_grud --real")
    if grud.get("split_mode"):
        row["grud_split_mode"] = grud["split_mode"]
    row["primary_metrics"] = ["pr_auc", "brier", "net_benefit_test_at_operating"]
    row["roc_auc_is_secondary"] = True
    row["h1_note"] = (
        "Prefer deathtime-v2 Brier + net benefit at operating threshold; "
        "do not roll back to dod solely to inflate PR-AUC."
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="", help="optional JSON path under artifacts/")
    args = parser.parse_args()
    result = compare()
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
