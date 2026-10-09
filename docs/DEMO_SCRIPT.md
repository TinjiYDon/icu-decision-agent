# 答辩演示口播（3–5 分钟）

> 启动：`.\scripts\run_console.ps1` → http://localhost:8501  
> 前置：已 restore S2 dump（feat≈472k），且存在 `artifacts/models/lgbm_mortality_12h.txt`  
> **叙事骨架**：[MASTER_NARRATIVE.md](MASTER_NARRATIVE.md) §6  
> **DCA**：优先已生成的 `artifacts/dca/dca_lgbm_*.json`（`python -m scripts.d2_dca_full --model lgbm`）；无报告时验收页回退抽样曲线。

## 流程

1. **项目（总览）**（约 45s）  
   - **总目标**：可决策的早期预警 CDSS（不是刷 AUC）。  
   - 问题：入科后 `h∈{0,1,2,4,6}`，预测未来 12h 死亡风险。  
   - 指主指标条：样本 ~472k、**PR-AUC / Brier / 工作点**；按小时 PR-AUC 小图。  
   - 一句免责：研究演示，非床旁器械。

2. **监测**（约 2min）  
   - 侧栏确认 `stay=` / `h=` 会随选择变化（ui v4.1）。  
   - 点「高风险样例」→ 风险徽章 + 建议动作 + 多时刻曲线。  
   - 面板以 **年龄/化验** 为主；心率等缺测显示 `—`（dump 口径，见 [`DUMP_READY.md`](DUMP_READY.md)）。  
   - 指 SHAP：优先解释**实际有值**特征（勿用全零向量讲故事）。  
   - 再点「低风险样例」对比。

3. **验收 + DCA 口述**（约 1min）  
   - Layer1 行数门禁（约 472k，五时刻）。  
   - 校准曲线：看预测概率与实际阳性率是否同向，不讲「绝对校准完美」。  
   - 切到 **决策净受益** 图（完整协议见 `artifacts/dca/`，否则为测试集抽样 ≤5k）：  
     - **三条线**：模型净受益 vs「全部干预」vs「全不干预」。  
     - **怎么读**：预警阈值 \(t\) 上，模型曲线高于两条基线，才谈该阈值有决策价值。  
     - **代价比**：默认 FP:FN = 1:4（漏诊更贵），对应早期预警低阈值（约 0.05–0.15），**不要**用 0.5 当 ICU 工作点。  
     - **阴影带**：Bootstrap 95% CI；带过宽则只说趋势、不报精确净受益数字。  
     - **工作点**：指图上/文案中的 F1 或临床成本寻优阈值，报精确率、召回、净受益各一句。  
     - **稀有阳性**：NB 可能为负仍可「优于全部干预」；主报仍是 **PR-AUC / Brier**，DCA 只回答「在哪个阈值值得响」。  
   - 强调：勿单报 ROC。

4. **创新**（约 45s）  
   - 底座 LightGBM vs 切口 H1–H3；TFT-lite 只是对照消融。  
   - 有 `artifacts/d3/report.md` 则指 PR-AUC 主表；化验三方数字须为因果窗重跑后版本。

5. **收尾**（可选 30s）  
   - MCP：`predict_risk(stay_id, hour_index=None)`；调参页可改建议阈值。  
   - 解释页可点采纳/驳回/编辑 → HITL 飞轮（`summarize_hitl`），本仓不驱动床位调度。

## 一键启动

```powershell
cd d:\project\icu-decision-agent
.\scripts\run_console.ps1
```

若监测全是 0 / 占位警示：重新 restore S2 dump，**不要**再跑会清空 feat 的 P0 ETL。
