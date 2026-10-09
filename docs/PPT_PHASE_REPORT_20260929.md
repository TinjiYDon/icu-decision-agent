# ICU Decision Agent · 阶段性汇报大纲

> 受众：项目组 / 答辩预演 · 语言：中文 · 建议 10–12 页  
> 素材：`docs/STATUS.md` · `docs/PHASE_PROGRESS_20260929.md` · `docs/MERGE_GATE.md`  
> 导出：可用 OpenClaw `/ppt` 转 `.pptx`（本文件为大纲）

## 1. 封面
- 标题：ICU 预警——创新不在「又一个 LightGBM」
- 副题：H1 标签/DCA · H2 公平双轨 · H3 解释可信
- 日期：2026-09-29

## 2. 底座 vs 切口（老师第一问）
- 底座：MIMIC + 无泄漏特征 + LightGBM 默认推理
- 切口：deathtime v2、PR-AUC/Brier、DCA、GRU-D 同 split、SHAP→RAG→LLM

## 2. 问题定义
- 预测时刻 \(t=\mathrm{intime}+h\)，\(h\in\{0,1,2,4,6\}\)
- 标签：未来 12h 死亡（deathtime v2）
- **独立 CDSS**；不做床位调度

## 3. 数据与标签诚实性
- 472k 行 · 阳性率 ~1.19%（更稀更严）
- 不以刷高旧 dod 指标退回

## 4. 主模型成绩
- test ROC=0.776（对照）
- **PR-AUC=0.092 · Brier=0.151（主验收）**

## 5. 创新 DX-1 解释安全
- 引用校验 / RAG / 过承诺控制
- 50 stays 真实评测指标栈

## 6. 创新 DX-2 DCA
- 决策曲线支撑阈值权衡口述
- Streamlit 接受/图表页

## 7. 创新 DX-3 时序双轨
- GRU-D vs LGBM 公平对照
- 主表看 PR-AUC/Brier（非只刷 ROC）

## 8. 架构 ADR-001
- UI → L4 `ui_queries`；禁 Streamlit 直连 SQL
- PR#19 已合入

## 9. 合入与清理
- 合入：#18 DX · #19 ADR
- 关闭：#12/#15–17 chore（无创新产出）
- 当前 open PR=0

## 10. 与 scheduling 的边界
- 两仓无代码依赖
- 叙事分离：预警 vs 床位运筹

## 11. 下一步
- HITL KPI 写入 STATUS
- Demo 挂 DCA 口述；可选 dump 同步

## 12. Q&A
- 预留
