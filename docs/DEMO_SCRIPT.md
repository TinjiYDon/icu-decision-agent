# 答辩演示口播（3–5 分钟）

> 启动：`.\scripts\run_console.ps1` → http://localhost:8501  
> 前置：已 restore S2 dump（feat≈472k），且存在 `artifacts/models/lgbm_mortality_12h.txt`

## 流程

1. **项目（总览）**（约 45s）  
   - 问题：入科后 `h∈{0,1,2,4,6}`，预测未来 12h 死亡风险。  
   - 指主指标条：样本 ~472k、**PR-AUC / Brier / 工作点**；按小时 PR-AUC 小图。  
   - 一句免责：研究演示，非床旁器械。

2. **监测**（约 2min）  
   - 侧栏确认 `stay=` / `h=` 会随选择变化（ui v4.1）。  
   - 点「高风险样例」→ 风险徽章 + 建议动作 + 多时刻曲线。  
   - 面板以 **年龄/化验** 为主；心率等缺测显示 `—`（dump 口径，见 [`DUMP_READY.md`](DUMP_READY.md)）。  
   - 指 SHAP：优先解释**实际有值**特征（勿用全零向量讲故事）。  
   - 再点「低风险样例」对比。

3. **验收**（约 1min）  
   - Layer1 行数门禁（约 472k，五时刻）。  
   - 校准曲线 + **决策净受益**曲线（测试集抽样 ≤5k）。  
   - 强调：稀有阳性优先 PR-AUC，勿单报 ROC。

4. **收尾**（可选 30s）  
   - MCP：`predict_risk(stay_id, hour_index=None)`；调参页可改建议阈值。

## 一键启动

```powershell
cd d:\project\icu-decision-agent
.\scripts\run_console.ps1
```

若监测全是 0 / 占位警示：重新 restore S2 dump，**不要**再跑会清空 feat 的 P0 ETL。
