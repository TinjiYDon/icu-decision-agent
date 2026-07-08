# 任务 Backlog（垂直切片）

复制到 GitHub Issue，标签见 CONTRIBUTING.md。

---

## S0 · 数据检查点 ✓

- [x] #S0-1 ETL 94k stays + dump + 冒烟（成员 A）

---

## S1 · 训练闭环

- [ ] **#S1-1** 成员 B：`application.train` 全量跑通，AUC/阳性率写入 STATUS  
  - 验证：`python -m application.train`  
  - 测试：新增 `tests/test_train.py`（smoke：能 fit、能 load）

- [ ] **#S1-2** 成员 A：确认 `feat.sample_matrix` / `label.outcomes` 与 B 对齐  
  - 验证：`pytest tests/test_etl.py -q`

---

## S2 · 单患者预测

- [ ] **#S2-1** 成员 B：`domain/models` 暴露 `predict(stay_id) -> score`  
- [ ] **#S2-2** 成员 C：`application/predict_patient.py`（或扩展 train 模块）编排 L2→L3  
  - 验证：固定 demo stay_id 输出可复现  
  - 测试：`tests/test_predict.py`

---

## S3 · Streamlit 演示

- [ ] **#S3-1** 成员 C：`presentation/streamlit_app.py` 选 stay → 风险分 + Top 因素  
  - 验证：`streamlit run presentation/streamlit_app.py`  
  - 禁止：页面内 SQL / 直接 import domain

---

## S4 · MCP（P2，可选）

- [ ] **#S4-1** 成员 C：MCP server 包装 L4 `predict_risk`  
- [ ] **#S4-2** 成员 B：JSON schema 文档写入 INNOVATION_ROADMAP

---

## 基础设施（贯穿）

- [ ] **#INF-1** 成员 A：CI 跑 pytest（GitHub Actions 可选）  
- [ ] **#INF-2** 成员 C：PR 合并 checklist 执行
