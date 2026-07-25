# 参数与数据故事（人机可读）

> Owner：C 维护 · B 补全训练指标 · A 补全 dump  
> 更新：2026-07-25 · Wave1：无泄漏特征 + 真三集划分

## 人读摘要

| 项 | 内容 |
|----|------|
| 标签 | `mortality_12h`：入科后 12h 内死亡 |
| 特征 | 仅入科可知（见 `configs/features.yaml`） |
| 划分 | train/val/test = 0.7/0.1/0.2 · stay_id · seed=42 |
| Wave2 | B 全量重训后写 `auc_val` / `auc_test` 到 STATUS |

## 特征（P0 Wave1）

| 特征 | 含义 | 入科可知？ |
|------|------|------------|
| `anchor_age` | 年龄 | ✅ |
| `gender_m` | 男性=1 | ✅ |
| `careunit_*` | 入科科室 one-hot | ✅ |
| ~~`los_hours`~~ | 总 LOS | ❌ 已剔除（泄漏） |
| ~~`hospital_expire_flag`~~ | 院内死亡 | ❌ 已剔除（泄漏） |

## 建议档位

| band | 阈值 | 含义 |
|------|------|------|
| observe | <0.2 | 观察 |
| recheck | <0.4 | 复查 |
| monitor | <0.7 | 加强监护 |
| escalate | ≥0.7 | 升级处置 |

## Agent 上下文

```text
契约：configs/features.yaml · configs/labels.yaml split
划分：domain/models/split.py → artifacts/models/split_manifest_mortality_12h.json
训练：python -m application.train（须先 build_features；旧 artifact 不兼容）
禁止：FEATURE_COLS 加回 hospital_expire_flag / 总 los_hours
验收：pytest tests/test_features_leak.py tests/test_predict.py -q
```
