# 任务 Backlog（垂直切片）

复制到 GitHub Issue，标签见 CONTRIBUTING.md。

---

## S0 · 数据检查点 ✓

- [x] #S0-1 ETL 94k stays + dump + 冒烟（成员 A）

---

## S1 · 训练闭环

- [ ] **#S1-1** 成员 B：`application.train` 全量跑通，`auc_val`/`auc_test`/阳性率写入 STATUS（**Wave2**）
- [ ] **#S1-2** 成员 A：feat/label + 非 schemas_only dump
- [x] **#S1-3** 成员 C：去除泄漏特征 + 真三集划分（**Wave1** ✅ 2026-07-25）

---

## S2 · 单患者预测

- [x] **#S2-1** 成员 B：`domain/models/lgbm.py` · `predict_stay(stay_id)` ✅（C 联调落地）
- [x] **#S2-2** 成员 C：`application/predict_patient.py` L4 + 缓存 ✅
- [x] **#S2-2b** SHAP/booster 缓存、`score_kind` 展示（C 加固 ✅）

---

## S3 · Streamlit 演示

- [x] **#S3-1** 成员 C：`presentation/streamlit_app.py` 选 stay → 风险分 + Top 因素 ✅
  - 验证：`streamlit run presentation/streamlit_app.py`
  - 禁止：页面内 SQL / 直接 import domain

---

## S4 · MCP（P2）

- [x] **#S4-1** 成员 C：MCP server 包装 L4 `predict_risk`（骨架 ✅ 2026-07-22）  
- [ ] **#S4-2** 成员 B：JSON schema 文档写入 INNOVATION_ROADMAP（可对照 `presentation/mcp_tools.py` PREDICT_RISK_SCHEMA）

---

## 基础设施（贯穿）

- [ ] **#INF-1** 成员 A：CI 跑 pytest（GitHub Actions 可选）  
- [ ] **#INF-2** 成员 C：PR 合并 checklist 执行
