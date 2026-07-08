# Restore dump → optional ETL (dump data not in GitHub)
param(
    [string]$DumpFile = "",
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

if (-not $DumpFile) {
    $latest = Get-ChildItem (Join-Path $root "dumps") -Filter "icu_decision_P0-etl_*.dump" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) { $DumpFile = $latest.FullName }
    else { throw "No dump in dumps/. Pass -DumpFile or run export_layer1.ps1 first." }
}

if (-not $SkipRestore) {
    & (Join-Path $PSScriptRoot "restore_layer1.ps1") -DumpFile $DumpFile -PgHost $PgHost -PgPort $PgPort
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

if (-not $SkipEtl) {
    $env:PYTHONPATH = $root
    Write-Host "Running ETL..."
    & $py -m application.etl_pipeline
}

Write-Host "Done. Next: python -m application.run_p0  (with PYTHONPATH=$root)"
