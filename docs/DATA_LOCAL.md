# 本地数据与 ETL（不入 GitHub）

## SSOT 路径

```
d:\project\_local-data\mimic\
├── mimic-iv-3.1.zip
├── extracted\mimic-iv-3.1\
├── icu_decision_P0-etl_mimic_94458stays_20260708.dump   ← 最新 Layer1（含 ETL）
├── icu_decision_P0-schema-only_legacy_20260614.dump     ← 旧版仅 schema
└── import_layer0.ps1
```

**命名规则：** `{db}_P0-etl_{layer0}_{N}stays_{yyyymmdd}.dump`

## 配置（本地 gitignore）

`configs/database.yaml`：

```yaml
layer0:
  mimic_source: mimic   # demo | full | mimic
  mimic_dsn: "postgresql+psycopg://icu_dev:icu_dev@localhost:5432/mimic"
```

`configs/data.yaml`：`source: mimic`

## 负责人：生产 dump

```powershell
$env:PYTHONPATH = (Get-Location)
.\scripts\run_data_pipeline.ps1
# 或分步：run_etl_stage → export_layer1.ps1 -Target decision -SchemasOnly -MimicSource mimic
```

输出：`dumps/` + 镜像 `_local-data/mimic/`

## 队友：restore + 可选 ETL

```powershell
.\scripts\restore_layer1.ps1 -Target decision `
  -DumpFile "d:\project\_local-data\mimic\icu_decision_P0-etl_mimic_94458stays_20260708.dump"
# 若 dump 已含 staging 数据可跳过 ETL；否则：
.\scripts\bootstrap_from_dump.ps1 -SkipRestore -MimicSource mimic
```

## 下一阶段（不在本仓库数据检查点内）

```powershell
python -m application.train
python -m application.run_p0
```
