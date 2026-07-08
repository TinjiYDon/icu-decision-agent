# 从 _local-data/mimic 的 Layer1 dump 恢复 → 跑 ETL（dump/ETL 数据均不入 GitHub）
# 用法：
#   .\scripts\bootstrap_from_dump.ps1
#   .\scripts\bootstrap_from_dump.ps1 -MimicSource demo -SkipRestore
param(
    [string]$DumpFile = "d:\project\_local-data\mimic\icu_decision_P0-etl_mimic_94458stays_20260708.dump",
    [ValidateSet("demo", "full", "mimic")]
    [string]$MimicSource = "demo",
    [switch]$SkipRestore,
    [switch]$SkipEtl,
    [string]$PgHost = "localhost",
    [int]$PgPort = 5432
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

if (-not $SkipRestore) {
    & (Join-Path $PSScriptRoot "restore_layer1.ps1") `
        -Target decision `
        -DumpFile $DumpFile `
        -PgHost $PgHost `
        -PgPort $PgPort
}

$dataYaml = Join-Path $root "configs\data.yaml"
if (-not (Test-Path $dataYaml)) {
    Copy-Item (Join-Path $root "configs\data.yaml.example") $dataYaml
}
$dbYaml = Join-Path $root "configs\database.yaml"
if (-not (Test-Path $dbYaml)) {
    Copy-Item (Join-Path $root "configs\database.yaml.example") $dbYaml
}

Write-Host "Layer0 mimic_source -> $MimicSource (edit configs/database.yaml if needed)"
Write-Host "data.yaml source should be: mimic"

if (-not $SkipEtl) {
    $env:PYTHONPATH = $root
    Write-Host "Running ETL..."
    & $py -m application.etl_pipeline
}

Write-Host "Done. Next: python -m application.run_p0  (with PYTHONPATH=$root)"
