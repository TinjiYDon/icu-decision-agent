# 总纲领 · 一条线串起全仓（2026-11-21 · V3 升级版）

> **详细公式与技术规范** → [OPTIMIZATION_SPEC.md](OPTIMIZATION_SPEC.md)  
> **阶段 0–5 行动清单** → [ACTION_CHECKLIST.md](ACTION_CHECKLIST.md) · **唯一数字台账** → [METRICS_LEDGER.md](METRICS_LEDGER.md) · **阶段门禁** → [STAGE.md](STAGE.md)

---

## 1. 总目标（North Star）

> 在严格时间因果（化验 `storetime <= t` · 在院存活过滤 · `subject_id` 患者级切分）下，为 ICU 入科早期提供**事件级目标可优化、有限预算队列可动态决策、标签时间精度可稳健归因、解释与事实纠正可人审留痕**的 12 小时院内死亡预警决策系统。

### 三大核心支柱（正面回应中期导师三问）

| 支柱 | 回应痛点 | 核心机制 |
| :--- | :--- | :--- |
| **支柱一：状态依赖动态复核队列** | 只做了预警，决策还没做 | 维护患者历史提醒状态 $\mathbf{z}_{s,h}=(I,\tau^{\mathrm{last}},p^{\mathrm{last}})$，在有限预算 $K$ 下对比 D0（直选）/ D1（静默排重）/ D2（跃升再报）/ D2-Dual，统一以**独立事件首次覆盖率、提前时长与 Event-NNR** 评价 |
| **支柱二：LightGBM 事件级目标改写** | LightGBM 只是调包 | **M0（无权重强基线）→ M1（事件/提前量归一）→ M2a（事件级 Smooth-Max + BCE 自定义 `fobj`，解析梯度 $g$ + 严格正定凸 MM 海森 $h$）** + 开发集概率校准 |
| **支柱三：同源粗化 LARE + $\varepsilon$-近优归因** | 公式改进与创新点不够清晰 | 在多自由度决策空间上做 1h/6h/12h/日级同源粗化与同翻转率随机对照，用 $\varepsilon$-近优策略集划分共识/争议复核集，以 $D_{\mathrm{cross}}$ vs $D_{\mathrm{within}}$ 严格分离标签粗化与近优多重性 |

**明确裁撤与边界**：旧 H4 效用阈值 / H5 稀疏融合 / H6 CUGEW 已彻底裁撤并归档至 `docs/archive/`；裁撤 M2b 成对排序；遮蔽回填与事实纠正收归 Streamlit 软著原型（VoI 演示），不冒充真人医生实验；不重启科室层级床位调度。

---

## 2. 端到端闭环流水线

```text
MIMIC-IV v3.1（storetime <= t · 在院存活 · subject_id 60/20/20 · 精确 deathtime）
  → LightGBM M0 → M1 → M2a（事件级 LSE+BCE 自定义 fobj）+ 概率校准（Brier / ECE）
  → 动态复核队列（预算 K · D0 / D1 / D2 / D2-Dual）→ 事件级首次覆盖 / 提前时长 / Event-NNR
  → 同源时间粗化（1h/6h/12h/日级）+ 多自由度 ε-近优共识/争议名单归因
  → Streamlit 原型（SHAP→RAG→LLM · 带时间/来源的事实纠正与双快照留痕）
```

---

## 3. 3 分钟答辩故事线

1. **痛点**：ICU 医护复核精力稀缺，静态逐行风险打分既让同一危重患者反复占额，又使标准 LightGBM 浪费容量在已显性旧病例上，同时历史死亡记录的时间粗化会扭曲复核规则选择。
2. **底座**：先补齐体征并封版严格因果底座（`storetime <= t`、在院过滤、`subject_id` 切分、去 `scale_pos_weight` 校准）。
3. **三层创新**：用状态依赖队列补齐**决策层**；用解析梯度与凸 MM 海森的 M2a 自定义目标升级 **LightGBM 模型层**；用同源粗化 + 多自由度 $\varepsilon$-近优集合讲透**经管资源配置稳健性**。
4. **验证与边界**：1,000 次患者级整组 Bootstrap 严格检验，事实纠正作为软著系统原型留痕，不夸大真人临床疗效。
