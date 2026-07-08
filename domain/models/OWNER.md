# 负责人：成员 B（算法）

- **职责**：LightGBM 训练/推理、artifacts、`model.registry`
- **入口**：`domain/models/lgbm.py` · `application/train.py`
- **输出契约**：predict 返回 score + 可选 SHAP 因子（供 C 的 L4 使用）
