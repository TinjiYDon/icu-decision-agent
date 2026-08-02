# mortality_12h 标签审计

> 更新：2026-07-31 · 算法复核结论

## 当前实现

`domain/labels/mortality_12h.py` 使用 ICU `intime` 与 `patients.dod` 判断入科后
12 小时内死亡。若 `dod` 是日期类型，代码把整天视为一个死亡时间区间。

## 风险

1. `patients.dod` 通常只有日期精度，不能精确判断 12 小时时窗。
2. 配置写 `icu_death`，注释又写“ICU/院内死亡”，目标定义不唯一。
3. 当前 full dump 只含已经生成的标签，没有 `dod/deathtime`，无法从 dump 内量化误标率。
4. 当前标签逻辑没有明确要求死亡发生在 ICU stay 内。

因此，现有 2,099 个阳性可用于工程基线复现，但标签尚未达到临床研究口径。

## 修正验收口径

数据负责人提供 Layer0 精确时间字段后，算法负责人应：

1. 将目标明确为“ICU 入科后 12h 内且 ICU stay 内死亡”或“院内 12h 死亡”，二选一。
2. 优先使用 `admissions.deathtime` 等精确时间戳；不得用日期级 `dod` 伪装成精确时间。
3. 报告新旧标签的变化数、阳性率与对 AUC/PR-AUC 的影响。
4. 固化标签版本，并重新生成 full dump、split manifest 和基线报告。

在此之前，不应把当前模型描述为经过临床验证的 12h 死亡预警模型。
