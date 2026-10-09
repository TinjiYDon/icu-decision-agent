# 创新架构 · 新旧对比 · 算法与数学公式

> 与 [MASTER_NARRATIVE.md](MASTER_NARRATIVE.md) 配套 · 2026-10-09  
> **说明**：骨干分类器仍为 LightGBM；「架构创新」指 **决策—融合—可信交付的系统架构与机制层**，不是新深度网络拓扑。

---

## 1. 创新系统架构（总图）

记 ICU stay 为 \(s\)，预测时刻 \(t = \mathrm{intime}(s) + h\)，\(h \in \mathcal{H}=\{0,1,2,4,6\}\)。

```text
                    x_<t (因果特征)          y_[t,t+H] (12h 死亡标签)
                           \                      /
                            \                    /
                             v                  v
                        +-----------+    +-------------+
                        |  LightGBM |--->|  p(s,h)     |  骨干风险分
                        +-----------+    +------+------+
                                                |
                 +------------------------------+------------------------------+
                 |                              |                              |
                 v                              v                              v
        +----------------+            +------------------+            +----------------+
        | H5 稀疏门控融合 |            | H4 效用阈值选择  |            | TreeSHAP 归因  |
        | p_fuse(s,h)    |            | τ* = argmax NB   |            +--------+-------+
        +--------+-------+            +--------+---------+                     |
                 |                             |                               v
                 +--------------> 报警 a=1{p≥τ*} <-------------+     RAG + LLM 解释
                                   |                           |               |
                                   v                           |               v
                              care_plan 建议                    |          HITL 人审
                                   |                           |               |
                                   +------------ 飞轮回流 ------+---------------+
```

**相对「旧架构」的本质变化**：旧系统止于「打分 →（可选）F1 阈值 → 展示」；新系统在同一骨干上增加 **效用最优工作点（H4）**、**完备度门控融合（H5）** 与 **可审计解释环（H3）**，形成闭环。

---

## 2. 新旧架构对比

| 维度 | 旧流水线（基线 CDSS Demo） | 本仓创新架构（机制层） |
|------|---------------------------|------------------------|
| 问题定义 | 常单报 ROC / 粗死亡日 `dod` | 主报 PR-AUC / Brier / **净受益**；标签协议可升级 v2 |
| 时间因果 | 易隐含泄漏 | 硬约束 \(\mathrm{charttime} < t\) |
| 分数 | 单时刻 \(p(s,h)\) | \(p(s,h)\) + 多时刻轨迹 \(p^{\mathrm{traj}}\) |
| 工作点 | \(\tau_{\mathrm{F1}}=\arg\max_\tau F_1\) | \(\tau_{\mathrm{NB}}=\arg\max_\tau \mathrm{NB}\)（临床窗 + 报警率帽） |
| 多源信息 | 无，或两模型各训各测 | 同 stay/split；**稀疏门控融合** |
| 解释 | SHAP 条或自由文本 | SHAP→RAG→LLM + 引用校验 + HITL |
| 验收 | 单一 AUC | G1–G5 指标栈 + 可证伪负结果 |

---

## 3. 符号表

| 符号 | 含义 |
|------|------|
| \(s\) | ICU stay |
| \(h\) | 预测小时偏移，\(t=\mathrm{intime}+h\) |
| \(H\) | 标签地平线（本仓 \(H=12\) 小时） |
| \(\mathbf{x}_{s,<t}\) | \(t\) 之前可得特征 |
| \(y_{s,h}\in\{0,1\}\) | \([t,t+H]\) 内是否死亡 |
| \(p(s,h)\in[0,1]\) | LightGBM 输出的风险概率 |
| \(\tau\) | 报警阈值；\(a=\mathbf{1}\{p\ge\tau\}\) |
| \(\lambda\) | FP:FN 代价比（默认 \(\lambda=4\)） |
| \(c(s,h)\in[0,1]\) | 特征完备度（非空比例） |
| \(\gamma\ge 0\) | 门控幂指数（实现中 `gate_power`） |

---

## 4. 数据与因果层

### 4.1 合法样本

\[
\mathcal{D}=\bigl\{\,(s,h,\mathbf{x}_{s,<t},\,y_{s,h})\;\big|\; h\in\mathcal{H},\ \mathbf{x}\text{ 仅含 }\mathrm{charttime}<t\,\bigr\}.
\]

### 4.2 标签（12h 死亡）

\[
y_{s,h}=\mathbf{1}\{\exists\,\text{死亡时间 }d:\ t\le d < t+H\}.
\]

（实现上可优先 `deathtime`，否则回退 `dod`；机制实验主表可固定旧 S2 口径。）

### 4.3 Stay 级划分

按 \(s\) 分层划分 train/val/test，禁止同一 stay 跨集合——保证多时刻行不泄漏划分。

---

## 5. 骨干风险模型（非创新卖点，但是底座）

LightGBM 学习条件概率的可校准打分函数：

\[
p(s,h)=\widehat{\Pr}\bigl(y_{s,h}=1\,\big|\,\mathbf{x}_{s,<t}\bigr)=f_{\mathrm{LGBM}}(\mathbf{x}_{s,<t}).
\]

个体归因采用 TreeSHAP。设特征 \(j\) 的 Shapley 值 \(\phi_j\)，满足

\[
f(\mathbf{x})=\phi_0+\sum_j\phi_j.
\]

---

## 6. H4 · 效用校准工作点（UCEW）——核心算法

### 6.1 广义净受益（Decision Curve）

在阈值 \(\tau\)、代价比 \(\lambda\) 下，令

\[
\mathrm{odds}(\tau)=\frac{\tau}{1-\tau},\quad
n=|\mathcal{V}|,\quad
\mathrm{TP}=\sum_i\mathbf{1}\{p_i\ge\tau,\,y_i=1\},\quad
\mathrm{FP}=\sum_i\mathbf{1}\{p_i\ge\tau,\,y_i=0\}.
\]

模型净受益与「全部干预」基线：

\[
\begin{aligned}
\mathrm{NB}_{\mathrm{model}}(\tau;\lambda)
&=\frac{\mathrm{TP}}{n}-\frac{\mathrm{FP}}{n}\cdot\lambda\cdot\mathrm{odds}(\tau),\\[4pt]
\mathrm{NB}_{\mathrm{all}}(\tau;\lambda)
&=\pi-(1-\pi)\cdot\lambda\cdot\mathrm{odds}(\tau),\quad
\pi=\tfrac{1}{n}\sum_i y_i,\\[4pt]
\mathrm{NB}_{\mathrm{none}}&=0.
\end{aligned}
\]

当 \(\lambda=1\) 时退化为经典 DCA；\(\lambda>1\) 表示漏诊更昂贵。

### 6.2 旧策略：验证集最大 \(F_1\)

\[
\tau_{\mathrm{F1}}=\arg\max_{\tau\in(0,1)} F_1(\tau)
=\arg\max_{\tau}\frac{2\,\mathrm{Prec}(\tau)\,\mathrm{Rec}(\tau)}{\mathrm{Prec}(\tau)+\mathrm{Rec}(\tau)}.
\]

### 6.3 新策略：约束最大净受益

设临床阈值窗 \([\tau_{\min},\tau_{\max}]\)（默认 \([0.05,0.40]\)）与报警率上限 \(\alpha\)（默认 \(0.20\)）：

\[
\begin{aligned}
\tau_{\mathrm{NB}}=\arg\max_{\tau}\ &\mathrm{NB}_{\mathrm{model}}(\tau;\lambda)\\
\text{s.t.}\quad
&\tau\in[\tau_{\min},\tau_{\max}],\\
&\frac{1}{n}\sum_i\mathbf{1}\{p_i\ge\tau\}\le\alpha.
\end{aligned}
\]

**与旧区别**：旧方法优化判别调和均值；新方法在可接受报警负荷内直接优化临床效用。  
**方法学注记**：若去掉窗与 \(\alpha\)，稀有事件下解常塌到极低 \(\tau\)（≈ treat-all）；故约束是架构的一部分，不是调参细节。

### 6.4 报警决策

\[
a(s,h)=\mathbf{1}\bigl\{p_{\mathrm{use}}(s,h)\ge\tau^\*\bigr\},
\quad
\tau^\*\in\{\tau_{\mathrm{F1}},\tau_{\mathrm{NB}}\}.
\]

其中 \(p_{\mathrm{use}}\) 可为骨干 \(p\) 或 H5 融合后的 \(p_{\mathrm{fuse}}\)。

**代码**：`domain/models/utility_calibrate.py` · `scripts/h4_utility_compare.py`。

---

## 7. H5 · 稀疏门控双轨融合（SGDF）——核心算法

### 7.1 当前时刻轨与轨迹轨

\[
p^{\mathrm{cur}}(s,h)=p(s,h),
\qquad
p^{\mathrm{traj}}(s,h)=\frac{1}{|\mathcal{H}_{\le h}|}
\sum_{h'\in\mathcal{H},\,h'\le h}p(s,h').
\]

（无 Layer0 时，\(p^{\mathrm{traj}}\) 为 **多时刻表格概率轨迹代理**；有真序列时可将 \(p^{\mathrm{traj}}\) 换为 GRU-D 输出，**门控公式不变**。）

### 7.2 完备度与门控

特征集 \(\mathcal{F}\)，对样本 \((s,h)\)：

\[
c(s,h)=1-\frac{1}{|\mathcal{F}|}\sum_{j\in\mathcal{F}}\mathbf{1}\{x_j\text{ 缺失}\}\in[0,1].
\]

门控权重（实现：`w = c**gate_power`）：

\[
w(s,h)=c(s,h)^{\gamma},\quad\gamma\ge 0.
\]

### 7.3 融合公式（新旧对照）

| 策略 | 公式 | 角色 |
|------|------|------|
| 旧·单时刻 | \(p=p^{\mathrm{cur}}\) | 基线 |
| 旧·无门控平均 | \(p=\tfrac12 p^{\mathrm{cur}}+\tfrac12 p^{\mathrm{traj}}\) | 忽视缺测 |
| **新·稀疏门控** | \(p_{\mathrm{fuse}}=(1-w)\,p^{\mathrm{cur}}+w\,p^{\mathrm{traj}}\) | 缺测→信当前；密→信轨迹 |

直觉：\(c\to 0\Rightarrow w\to 0\Rightarrow p_{\mathrm{fuse}}\to p^{\mathrm{cur}}\)；\(c\to 1\Rightarrow w\to 1\Rightarrow p_{\mathrm{fuse}}\to p^{\mathrm{traj}}\)。

**代码**：`domain/models/sparsity_fusion.py` · `scripts/h5_sparsity_fusion.py`。

---

## 8. H3 · 解释—人机环（结构式，非新网络）

设 SHAP 结构化因子集合 \(\Phi=\{\phi_j\}\)，检索证据集 \(\mathcal{R}\)，生成解释文本 \(e=g_{\mathrm{LLM}}(\Phi,\mathcal{R})\)。引用校验器 \(V(e,\mathcal{R})\in[0,1]\)（数值容差命中率等）。人决策

\[
d\in\{\mathrm{accept},\mathrm{reject},\mathrm{edit}\}
\]

写入审计日志，供阈值/知识回流。相对旧「只展示 SHAP 条」，新架构把 **可验证引用 + 可驳回** 做成闭环节点。

---

## 9. 主验收指标（数学定义）

对测试集预测概率 \(\hat p_i\) 与标签 \(y_i\)：

\[
\begin{aligned}
\mathrm{PR\text{-}AUC}&=\int_0^1\mathrm{Prec}(r)\,\mathrm{d}r
\quad\text{（平均精度）},\\[4pt]
\mathrm{Brier}&=\frac{1}{n}\sum_i(\hat p_i-y_i)^2,\\[4pt]
\mathrm{ROC\text{-}AUC}&=\Pr(\hat p^+>\hat p^-)
\quad\text{（仅对照）}.
\end{aligned}
\]

工作点层面报告 \(\mathrm{NB}_{\mathrm{model}}(\tau^\*;\lambda)\) 与报警率

\[
\mathrm{AlertRate}=\frac{1}{n}\sum_i\mathbf{1}\{\hat p_i\ge\tau^\*\}.
\]

---

## 10. 端到端推理算法（伪代码）

```text
Input: stay s, hour h, cost_ratio λ, gate_power γ
1  x ← features with charttime < intime+h
2  p_cur ← LGBM(x)
3  p_traj ← mean{ LGBM at hours h' ≤ h }   # or GRU-D if Layer0
4  c ← feature_completeness(x)
5  w ← c^γ
6  p ← (1-w)*p_cur + w*p_traj              # H5
7  τ* ← τ_NB from validation (H4)          # frozen at deploy
8  alert ← 1{p ≥ τ*}
9  Φ ← TreeSHAP(x); e ← LLM(RAG(Φ)); d ← clinician feedback
10 return p, alert, e, d
```

---

## 11. 与「调包拼模型」指控的对应答复

| 指控 | 本架构回应 |
|------|------------|
| 只是 LightGBM | 骨干是底座；**可发表切口在 \(\tau_{\mathrm{NB}}\) 与 \(p_{\mathrm{fuse}}\) 机制** |
| GRU/TFT 凑数 | 明确为对照/消融；默认推理不替换 |
| 无公式 | 本文给出 NB、约束优化、门控融合闭式 |
| 无新旧对比 | §2 / §6.2–6.3 / §7.3 表格对照 |

---

## 12. 相关文件

| 内容 | 路径 |
|------|------|
| 总叙事 | [MASTER_NARRATIVE.md](MASTER_NARRATIVE.md) |
| 假设与主表数字 | [ACADEMIC_THESIS.md](ACADEMIC_THESIS.md) |
| H4 实现 | `domain/models/utility_calibrate.py` |
| H5 实现 | `domain/models/sparsity_fusion.py` |
| DCA | `domain/models/dca.py` |
