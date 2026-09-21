"""Summarize HITL explain audit JSONL for the decision data flywheel (D-FLY)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from domain.explain.hitl_audit import audit_path

ROOT = Path(__file__).resolve().parents[1]


def _parse_ts(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def summarize_hitl(*, days: int | None = 7, log_path: Path | None = None) -> dict[str, Any]:
    path = log_path or audit_path()
    if not path.is_file():
        return {
            "status": "empty",
            "path": str(path),
            "n": 0,
            "accept_rate": None,
            "reject_rate": None,
            "edit_rate": None,
            "by_decision": {},
            "by_recommend": {},
            "note": "no audit log yet — use Streamlit 解释页提交反馈",
        }

    cutoff = None
    if days is not None and days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(days))

    decisions: Counter[str] = Counter()
    recommends: Counter[str] = Counter()
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = _parse_ts(str(row.get("ts") or ""))
        if cutoff is not None and ts is not None and ts < cutoff:
            continue
        n += 1
        d = str(row.get("decision") or "unknown")
        decisions[d] += 1
        rec = row.get("recommend")
        if rec is not None:
            recommends[str(rec)] += 1

    def rate(key: str) -> float | None:
        return round(decisions[key] / n, 4) if n else None

    return {
        "status": "ok",
        "path": str(path),
        "window_days": days,
        "n": n,
        "by_decision": dict(decisions),
        "by_recommend": dict(recommends),
        "accept_rate": rate("accept"),
        "reject_rate": rate("reject"),
        "edit_rate": rate("edit"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reflow_hint": [
            "1_revise_rag_knowledge",
            "2_tune_recommend_thresholds",
            "3_manual_hard_cases_only",
            "never_auto_retrain_on_reject",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="HITL flywheel weekly summary")
    p.add_argument("--days", type=int, default=7, help="lookback days; 0 = all")
    p.add_argument("--out", type=str, default="", help="optional JSON output path")
    args = p.parse_args()
    days = None if args.days == 0 else args.days
    result = summarize_hitl(days=days)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
