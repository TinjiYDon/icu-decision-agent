"""端到端测试：调用 Agnes API 验证 SHAP+LLM 解释层整体逻辑。

用法：
    python scripts/test_explain_e2e.py
    python scripts/test_explain_e2e.py --stay-id 30031234 --hour 0
    python scripts/test_explain_e2e.py --mock  # 不查数据库，用 mock 数据测试
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domain.explain.config import get_config
from domain.explain.rag_retriever import is_available as rag_is_available, retrieve_context


# Mock 数据：模拟 L3 predict_stay() 的输出（用于无数据库环境测试）
MOCK_PREDICTION = {
    "stay_id": 30000001,
    "hour_index": 0,
    "status": "ok",
    "risk_score": 0.185,
    "score_kind": "probability",
    "recommend": {
        "band": "observe",
        "label": "观察（低风险）",
        "thresholds": {"observe": 0.2, "recheck": 0.4, "monitor": 0.7},
    },
    "top_factors": [
        {"feature": "lab_lactate", "value": 2.3, "shap": 0.0823},
        {"feature": "shock_index", "value": 1.1, "shap": 0.0456},
        {"feature": "vital_gcs_total", "value": 14, "shap": -0.0289},
        {"feature": "lab_ph", "value": 7.32, "shap": 0.0156},
    ],
    "features": {
        "lab_lactate": 2.3,
        "shock_index": 1.1,
        "vital_gcs_total": 14,
        "lab_ph": 7.32,
        "anchor_age": 68,
        "vital_temp": 36.8,
    },
}


def test_rag_retrieval() -> None:
    """测试 1：RAG 检索功能。"""
    print("\n[测试 1] RAG 检索")
    print("-" * 40)
    if not rag_is_available():
        print("  ⚠️ RAG 知识库不可用，跳过检索测试")
        return
    queries = [
        "乳酸 Lactate 2.3 mmol/L 临床意义",
        "休克指数 Shock Index 1.1 阈值",
        "GCS 评分 14 分 解读",
        "SOFA 评分标准",
    ]
    for q in queries:
        hits = retrieve_context(q, top_k=2)
        print(f"  查询: {q}")
        print(f"  命中: {len(hits)} 条")
        for h in hits[:2]:
            print(f"    - [{h.source}] {h.title} (score={h.score:.3f}, cat={h.category})")
        print()


def test_llm_api_connection() -> bool:
    """测试 2：Agnes API 连通性（最小调用）。"""
    print("\n[测试 2] Agnes API 连通性")
    print("-" * 40)
    cfg = get_config()
    if not cfg.llm.api_key:
        print(f"  ❌ AGNES_API_KEY 未配置")
        return False
    print(f"  Base URL: {cfg.llm.base_url}")
    print(f"  Model:    {cfg.llm.model}")
    print(f"  API Key:  {cfg.llm.api_key[:8]}...{cfg.llm.api_key[-4:]}")

    try:
        from domain.explain.llm_client import LLMClient
        client = LLMClient()
        resp = client.generate(
            system_prompt="你是一个测试助手，请回复「OK」。",
            user_prompt="请回复 OK",
            use_function_calling=False,
        )
        if resp.ok:
            print(f"  ✅ 调用成功，耗时 {resp.elapsed_ms}ms")
            print(f"  响应: {resp.content[:80]}")
            if resp.usage:
                print(f"  Token: {resp.usage}")
            return True
        else:
            print(f"  ❌ 调用失败: {resp.error}")
            return False
    except Exception as e:
        print(f"  ❌ 异常: {type(e).__name__}: {e}")
        return False


def test_full_explanation(stay_id: int | None = None, hour_index: int = 0, use_mock: bool = False) -> None:
    """测试 3：端到端解释生成。"""
    print("\n[测试 3] 端到端解释生成")
    print("-" * 40)
    from domain.explain.shap_llm import generate_explanation

    if use_mock or stay_id is None:
        print(f"  使用 Mock 数据 (stay_id={MOCK_PREDICTION['stay_id']})")
        pred = MOCK_PREDICTION
    else:
        print(f"  查询数据库 stay_id={stay_id}, hour={hour_index}")
        from application.predict_patient import predict_patient
        pred = predict_patient(stay_id, hour_index=hour_index)
        if pred.get("status") != "ok":
            print(f"  ❌ 预测失败: {pred.get('status')} - {pred.get('message', '')}")
            return

    print(f"  风险分数: {pred['risk_score']}")
    print(f"  风险分级: {pred['recommend']['label']}")
    print(f"  Top 因素: {len(pred['top_factors'])} 个")
    for f in pred["top_factors"]:
        print(f"    - {f['feature']}: value={f['value']}, shap={f['shap']}")

    print("\n  生成解释中...")
    result = generate_explanation(
        stay_id=pred["stay_id"],
        hour_index=pred["hour_index"],
        shap_output=pred["top_factors"],
        risk_score=pred["risk_score"],
        recommendation=pred["recommend"],
        features_display=pred.get("features"),
    )

    print(f"\n  状态: {result['status']}")
    print(f"  耗时: {result['elapsed_ms']}ms")
    print(f"  引用数: {len(result['references'])}")
    print(f"  降级模式: {result['structured'].get('fallback_mode', False)}")
    print(f"  RAG 可用: {result['structured'].get('rag_available', False)}")

    print("\n" + "=" * 60)
    print("解释报告（Markdown）")
    print("=" * 60)
    print(result["explanation"])

    print("\n" + "=" * 60)
    print("结构化输出（JSON）")
    print("=" * 60)
    # 简化输出，去掉过长的字段
    structured = result["structured"]
    print(json.dumps(structured, ensure_ascii=False, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description="SHAP+LLM 解释层端到端测试")
    parser.add_argument("--stay-id", type=int, default=None, help="ICU stay ID（不传则用 mock 数据）")
    parser.add_argument("--hour", type=int, default=0, help="预测时刻")
    parser.add_argument("--mock", action="store_true", help="强制使用 mock 数据")
    parser.add_argument("--skip-rag", action="store_true", help="跳过 RAG 检索测试")
    parser.add_argument("--skip-llm", action="store_true", help="跳过 LLM 连通性测试")
    args = parser.parse_args()

    print("=" * 60)
    print("SHAP+LLM 解释层 端到端测试")
    print("=" * 60)

    cfg = get_config()
    print(f"LLM 模型: {cfg.llm.model}")
    print(f"Embedding: {cfg.rag.embedding_model}")
    print(f"RAG 可用: {rag_is_available()}")

    if not args.skip_rag:
        test_rag_retrieval()

    if not args.skip_llm:
        llm_ok = test_llm_api_connection()
        if not llm_ok:
            print("\n⚠️ LLM 不可用，将只测试降级模式")

    test_full_explanation(stay_id=args.stay_id, hour_index=args.hour, use_mock=args.mock or args.stay_id is None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
