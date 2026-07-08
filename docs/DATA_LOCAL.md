# 本地数据说明

本文说明如何在本机准备 MIMIC Layer0 与 Layer1 dump。**数据文件不入 GitHub。**

## 目录约定

```
icu-decision-agent/
├── dumps/                          ← 导出/恢复的默认位置
│   └── icu_decision_P0-etl_*.dump
└── configs/
    ├── database.yaml               ← 本地创建（gitignore）
    └── data.yaml
```

## Layer0（MIMIC 源库）

在 `configs/database.yaml` 中配置：

```yaml
layer0:
  mimic_source: mimic   # demo | full | mimic
  mimic_dsn: "postgresql+psycopg://icu_dev:icu_dev@localhost:5432/mimic"
```

- `demo`：`mimic_iv_demo`（约 140 stays，快速验证）
- `mimic`：自建全量 `mimic` 库（需单独导入 MIMIC-IV，见 PhysioNet 许可）

`configs/data.yaml`：`source: mimic`

## 导出 dump

```powershell
$env:PYTHONPATH = (Get-Location)
.\scripts\run_data_pipeline.ps1
# 或分步：python -m application.run_etl_stage → .\scripts\export_layer1.ps1 -SchemasOnly
```

输出：`dumps/icu_decision_P0-etl_{layer0}_{N}stays_{date}.dump`

可选复制到网盘或共享目录（勿提交 Git）：

```powershell
.\scripts\run_data_pipeline.ps1 -DumpMirrorDir D:\backups\icu-decision
# 或：$env:LOCAL_DATA_MIRROR = "D:\backups\icu-decision"
```

## 从 dump 恢复

```powershell
.\scripts\restore_layer1.ps1 -DumpFile .\dumps\icu_decision_P0-etl_mimic_94458stays_20260708.dump
```

若 dump 已含 staging 数据，可跳过 ETL：

```powershell
.\scripts\bootstrap_from_dump.ps1 -SkipEtl
```

## 下一阶段

```powershell
python -m application.train
python -m application.run_p0
```
