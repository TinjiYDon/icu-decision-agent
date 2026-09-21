"""D1 解释链安全对照 · 批量评测 CLI。

用法：
    python scripts/audit_safety_metrics.py
    python scripts/audit_safety_metrics.py --stay-ids 30000153 30000213 --hours 1
    python scripts/audit_safety_metrics.py --mock --limit 10
    python scripts/audit_safety_metrics.py --out artifacts/explain_audit/report_20260914.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domain.explain.audit import (
    MODE_FULL,
    MODE_LLM_NO_RAG,
    MODE_RULE,
    MODES,
    run_audit_batch,
    summarize_report,
)


def _fetch_predictions(stay_ids: list[int], hour_index: int) -> dict[int, dict]:
    """从数据库获取预测结果（LGBM）。"""
    from application.predict_patient import predict_patient
    preds = {}
    for sid in stay_ids:
        try:
            pred = predict_patient(sid, hour_index=hour_index, model_type="lgbm")
            if pred.get("status") == "ok":
                preds[sid] = pred
        except Exception as e:
            print(f"  [warn] stay {sid} 预测失败: {e}")
    return preds


def _mock_predictions(stay_ids: list[int]) -> dict[int, dict]:
    """Mock 预测数据（不查数据库，不调 LLM 的 SHAP 部分）。"""
    import numpy as np
    rng = np.random.default_rng(42)
    preds = {}
    for sid in stay_ids:
        n = rng.integers(3, 6)
        feat_names = ["anchor_age", "lab_lactate", "vital_heart_rate", "lab_bun",
                       "gcs_total", "vasopressor_1h", "lab_potassium", "vital_temp"]
        top_factors = []
        for fn in feat_names[:n]:
            top_factors.append({
                "feature": fn,
                "value": round(float(rng.normal(0, 1)), 2),
                "shap": round(float(rng.normal(0, 0.1)), 4),
            })
        top_factors.sort(key=lambda x: abs(x["shap"]), reverse=True)
        preds[sid] = {
            "stay_id": sid,
            "hour_index": 1,
            "status": "ok",
            "risk_score": round(float(rng.uniform(0.05, 0.4)), 4),
            "score_kind": "probability",
            "recommend": {"band": "monitor" if rng.random() > 0.5 else "observe",
                          "label": "监测（中风险）" if rng.random() > 0.5 else "观察（低风险）"},
            "top_factors": top_factors,
            "features": {f["feature"]: f["value"] for f in top_factors},
        }
    return preds


def main() -> int:
    parser = argparse.ArgumentParser(description="D1 解释链安全对照批量评测")
    parser.add_argument("--stay-ids", type=int, nargs="+", default=[], help="指定 stay ID 列表")
    parser.add_argument("--hours", type=int, default=1, help="预测时刻 h")
    parser.add_argument("--limit", type=int, default=20, help="最多评测多少个 stay")
    parser.add_argument("--mock", action="store_true", help="使用 mock 数据（不查数据库）")
    parser.add_argument("--out", type=str, default="", help="输出 JSON 路径")
    parser.add_argument("--report", type=str, default="", help="输出 Markdown 报告路径")
    args = parser.parse_args()

    print("=" * 60)
    print("D1 解释链安全对照 · 批量评测")
    print("=" * 60)

    # 获取 stay IDs
    if args.stay_ids:
        stay_ids = args.stay_ids[:args.limit]
    elif args.mock:
        stay_ids = list(range(900001, 900001 + args.limit))
    else:
        # 从数据库取前 N 个 stay
        from infra.db import get_engine
        from sqlalchemy import text
        eng = get_engine()
        with eng.connect() as conn:
            rows = conn.execute(
                text("SELECT DISTINCT stay_id FROM label.mortality_12h "
                     "WHERE hour_index = :h ORDER BY stay_id LIMIT :n"),
                {"h": args.hours, "n": args.limit},
            ).mappings().all()
        stay_ids = [int(r["stay_id"]) for r in rows]

    if not stay_ids:
        print("错误：未找到可用 stay ID")
        return 1

    print(f"评测 stay 数：{len(stay_ids)}，时刻 h={args.hours}")
    print(f"模式：{' + '.join(MODES)}")
    print()

    # 获取预测结果
    if args.mock:
        print("  [mock] 使用 mock 预测数据...")
        predictions = _mock_predictions(stay_ids)
    else:
        print(f"  [db] 从数据库获取预测结果（h={args.hours}）...")
        predictions = _fetch_predictions(stay_ids, args.hours)

    if not predictions:
        print("错误：未获取到任何有效预测")
        return 1
    print(f"  有效预测：{len(predictions)} 条")
    print()

    # 跑评测
    def _pred_fn(sid: int, hour: int) -> dict:
        return predictions.get(sid, {})

    print("开始批量评测...")
    summary = run_audit_batch(
        stay_ids=list(predictions.keys()),
        hour_index=args.hours,
        prediction_fn=_pred_fn,
        limit=len(predictions),
    )

    # 输出 JSON
    out_path = Path(args.out) if args.out else (ROOT / "artifacts" / "explain_audit" / f"report_{Path(__file__).resolve().parents[1].stem}_{''.join(str(d) for d in __import__('datetime').date.today().timetuple()[:3])}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON 报告：{out_path}")

    # 输出 Markdown 报告
    report_md = summarize_report(summary)
    if args.report:
        rpt_path = Path(args.report)
    else:
        rpt_path = out_path.with_suffix(".md")
    rpt_path.write_text(report_md, encoding="utf-8")
    print(f"Markdown 报告：{rpt_path}")

    # 打印汇总
    print("\n" + "=" * 80)
    print("汇总指标（10项）")
    print("=" * 80)
    agg = summary["aggregate"]
    mode_labels = {MODE_RULE: "A. 规则模板", MODE_LLM_NO_RAG: "B. 单LLM无RAG", MODE_FULL: "D. SHAP+RAG+LLM"}
    for m in MODES:
        label = mode_labels.get(m, m)
        align = agg["shap_align"].get(m)
        ref = agg["ref_validity"].get(m)
        oc = agg["overcommit"].get(m)
        cov = agg["feature_coverage"].get(m)
        grd = agg["factual_grounding"].get(m)
        evi = agg["evidence_specificity"].get(m)
        rc = agg["risk_consistency"].get(m)
        td = agg["clinical_term_density"].get(m)
        rs = agg["readability_score"].get(m)
        def _s(v): return f"{v:.4f}" if v is not None else "—"
        print(f"  {label:20s} | SHAP={_s(align)} | 引效={_s(ref)} | 过度={_s(oc)} | 覆盖={_s(cov)} | ground={_s(grd)} | 具体={_s(evi)} | 一致={_s(rc)} | 术语={_s(td)} | 可读={_s(rs)}")

    print(f"\n耗时：{summary['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
