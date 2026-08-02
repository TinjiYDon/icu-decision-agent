# 答辩演示口播（3–5 分钟）

> 启动：`.\scripts\run_console.ps1` → http://localhost:8501  
> 前置：已 restore S2 dump，且存在 `artifacts/models/lgbm_mortality_12h.txt`

## 流程

1. **项目（总览）**（约 45s）  
   - 问题：入科后 `h∈{0,1,2,4,6}`，预测未来 12h 死亡风险。  
   - 指主指标条：样本 ~472k、**PR-AUC / Brier / 工作点**；按小时 PR-AUC 小图。  
   - 一句免责：研究演示，非床旁器械。

2. **监测**（约 2min）  
   - 侧栏点「高风险样例」→ 风险徽章 + 建议动作档位 + 多时刻曲线。  
   - 指 SHAP：为何此刻偏高。  
   - 再点「低风险样例」对比。

3. **验收**（约 1min）  
   - Layer1 行数门禁（约 472k）。  
   - 校准曲线 + **决策净受益**曲线（测试集抽样 ≤5k）。  
   - 强调：稀有阳性优先 PR-AUC，勿单报 ROC。

4. **收尾**（可选 30s）  
   - MCP：`predict_risk(stay_id, hour_index=None)`；调参页可改建议阈值。

## 一键启动

```powershell
cd d:\project\icu-decision-agent
.\scripts\run_console.ps1
```
