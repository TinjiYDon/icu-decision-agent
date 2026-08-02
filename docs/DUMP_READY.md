# Dump 交付说明（队友 restore / 训练）

> 更新：2026-08-02 · **仅保留 S2 full dump** · **不入 GitHub** · 线下单发

## 单发文件（Owner → 队友）

| 文件 | 绝对路径 | 说明 |
|------|----------|------|
| **主 dump** | `d:\project\icu-decision-agent\dumps\icu_decision_S2-full_mimic_94458stays_20260802.dump` | S2 · feat/label≈**472,290** · `h∈{0,1,2,4,6}` |
| 元数据（可选） | `d:\project\icu-decision-agent\dumps\DATA_VERSION_20260802_1758.json` | SHA / 行数说明 |

**SHA-256**：`5b71b752cb17bb8513c73682230329b1223a0baa7e52a791d3cd83e9409c1605`  
旧 Wave2（20260726）dump 已从本机清理；勿再索要旧文件作 S2 底座。

## 恢复 + 训练 + 监测台

```powershell
cd d:\project\icu-decision-agent
.\scripts\restore_layer1.ps1 -DumpFile .\dumps\icu_decision_S2-full_mimic_94458stays_20260802.dump
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m application.train --from-existing
.\.venv\Scripts\python.exe -m streamlit run presentation/streamlit_app.py --server.port 8501
```

验收：`SELECT COUNT(*) FROM feat.sample_matrix;` → ≈ **472290**

## 禁止

- 把 schemas_only / 旧单时刻 dump 当 S2 底座  
- restore 后无 Layer0 时跑默认 `application.train`（会重建并冲掉网格）  
- 宣称 dump 含 labevents / 把 dump 推进 GitHub  
