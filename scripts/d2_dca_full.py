"""D2 · 全量决策曲线分析（DCA）+ Bootstrap CI + 工作点。

用法：
    python -m scripts.d2_dca_full
    python -m scripts.d2_dca_full --model lgbm --cost-ratio 4 --bootstrap 1000
    python -m scripts.d2_dca_full --model grud --from-db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 确保项目根目录在 PYTHONPATH
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from domain.features.build import FEATURE_COLS
from domain.models.dca import run_dca
from domain.models.lgbm import _load_training_frame, _get_model_bundle
from domain.models.split import load_split_assignment


def _load_test_data(model: str = "lgbm") -> tuple[np.ndarray, np.ndarray, str]:
    """从 LGBM 测试集批量加载 y_true 和 y_prob（避免逐条 DB 连接）。

    关键优化：直接用 Booster 批量预测，不经过 predict_stay（每次 predict_stay
    会创建新 DB 连接并加载 SHAP，1000+ 次调用会耗尽连接池）。
    """
    if model == "lgbm":
        df = _load_training_frame()
        assignment = load_split_assignment()
        if not assignment:
            raise RuntimeError("无 split manifest，请先运行 python -m application.train")

        test_ids = [int(sid) for sid, split in assignment.items() if split == "test"]
        test_df = df[df["stay_id"].isin(test_ids)]
        if len(test_df) == 0:
            raise RuntimeError("test split 为空")

        # 批量预测（单次 Booster 调用，无需逐条 DB 连接）
        booster, _ = _get_model_bundle()
        X = test_df[FEATURE_COLS].to_numpy(dtype=float)
        prob = np.asarray(booster.predict(X), dtype=float)
        # 处理 raw score → probability
        if prob.min() < 0 or prob.max() > 1:
            prob = 1.0 / (1.0 + np.exp(-prob))
        y_true = test_df["label"].to_numpy(dtype=int)

        print(f"[D2] LGBM 测试集: n={len(y_true)}, 阳性率={y_true.mean():.2%}")
        return y_true, prob, "lgbm"

    else:
        raise ValueError(f"未知模型: {model}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="D2: 全量决策曲线分析（DCA）+ Bootstrap CI + 工作点",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python -m scripts.d2_dca_full                    # LGBM 默认参数
  python -m scripts.d2_dca_full --model lgbm       # LGBM 显式
  python -m scripts.d2_dca_full --model grud       # GRU-D（需先训练）
  python -m scripts.d2_dca_full --cost-ratio 3     # 调整 FP:FN 代价比
  python -m scripts.d2_dca_full --bootstrap 2000   # 更多 Bootstrap 样本
        """,
    )
    parser.add_argument("--model", choices=["lgbm", "grud"], default="lgbm",
                        help="模型类型（默认 lgbm）")
    parser.add_argument("--cost-ratio", type=float, default=4.0,
                        help="FP:FN 临床代价比（默认 4.0）")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Bootstrap 抽样次数（默认 1000）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--ci-level", type=float, default=0.95, help="置信水平（默认 0.95）")
    parser.add_argument("--no-save", action="store_true", help="不保存报告文件")
    args = parser.parse_args()

    print(f"[D2] 启动 DCA 分析")
    print(f"[D2] 模型={args.model}, 代价比={args.cost_ratio}:1, Bootstrap={args.bootstrap}")

    try:
        y_true, y_prob, model_name = _load_test_data(args.model)
        print(f"[D2] 数据加载完成: n={len(y_true)}, 阳性率={y_true.mean():.2%}")

        result = run_dca(
            y_true, y_prob,
            model_name=model_name,
            cost_ratio=args.cost_ratio,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
            save=not args.no_save,
        )

        # 打印摘要
        c = result.curve
        wp = c.working_point
        print()
        print("=" * 50)
        print("D2 DCA 结果摘要")
        print("=" * 50)
        print(f"  模型: {result.model_name}")
        print(f"  样本量: {c.n:,}")
        print(f"  阳性率: {c.prevalence:.2%}")
        print(f"  代价比 FP:FN: {c.cost_ratio}:1")
        print(f"  Bootstrap CI: {c.n_bootstrap} 次, 95%")
        print(f"  耗时: {c.elapsed_s:.1f}s")
        print()
        if wp:
            print("  工作点:")
            print(f"    阈值: {wp.threshold:.4f} ({wp.method})")
            print(f"    精确率: {wp.precision:.2%}")
            print(f"    召回率: {wp.recall:.2%}")
            print(f"    特异度: {wp.specificity:.2%}")
            print(f"    F1: {wp.f1:.4f}")
            print(f"    净受益: {wp.net_benefit:.4f}")
            print(f"    TP/FP/FN/TN: {wp.tp}/{wp.fp}/{wp.fn}/{wp.tn}")
        print()
        print("  关键阈值净受益:")
        for p in result.points:
            ci = ""
            if c.nb_model_ci_lower and c.nb_model_ci_upper:
                idx = next((i for i, t in enumerate(c.thresholds) if abs(t - p.threshold) < 0.001), None)
                if idx is not None:
                    ci = f"  [{c.nb_model_ci_lower[idx]:.4f}, {c.nb_model_ci_upper[idx]:.4f}]"
            print(f"    t={p.threshold:.2f}: NB={p.net_benefit_model:.4f}{ci}  (all={p.net_benefit_treat_all:.4f})")
        print("=" * 50)
        print(f"  报告已保存至: {result.report_path}")
        print()

        # 返回码
        if c.nb_model and c.nb_model_ci_lower:
            # 检查是否有净受益区间
            above_treat_none = any(nb > 0 for nb in c.nb_model)
            print(f"  结论: {'有净受益区间' if above_treat_none else '无显著净受益'}")
        sys.exit(0 if c.status == "ok" else 1)

    except Exception as exc:
        print(f"[D2] ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
