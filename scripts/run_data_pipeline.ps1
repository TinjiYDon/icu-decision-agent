# ETL → export Layer1 dump → smoke tests (stops before model training)
param(
    [switch]$SkipEtl,
    [switch]$SkipExport,
    [switch]$SkipTests,
    [string]$DumpMirrorDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$env:PYTHONPATH = $root

Write-Host "=== check_env ==="
& $py scripts\check_env.py
if ($LASTEXITCODE -ne 0) { throw "check_env failed" }

if (-not $SkipEtl) {
    Write-Host "=== ETL (checkpoint: before train) ==="
    & $py -m application.run_etl_stage
    if ($LASTEXITCODE -ne 0) { throw "ETL failed" }
}

if (-not $SkipExport) {
    Write-Host "=== export Layer1 dump ==="
    & (Join-Path $PSScriptRoot "export_layer1.ps1") -SchemasOnly -MimicSource mimic
    $mirror = if ($DumpMirrorDir) { $DumpMirrorDir } elseif ($env:LOCAL_DATA_MIRROR) { $env:LOCAL_DATA_MIRROR } else { "" }
    if ($mirror) {
        New-Item -ItemType Directory -Force -Path $mirror | Out-Null
        Get-ChildItem (Join-Path $root "dumps") -Filter "icu_decision_P0-etl_*.dump" |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
            ForEach-Object { Copy-Item $_.FullName $mirror -Force; Write-Host "Mirrored: $($_.Name) -> $mirror" }
    }
}

if (-not $SkipTests) {
    Write-Host "=== pytest (smoke/db/etl) ==="
    & $py -m pytest tests/test_smoke.py tests/test_db.py tests/test_etl.py -q
    if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
}

Write-Host ""
Write-Host "CHECKPOINT OK: ETL + dump + smoke. Next: python -m application.train"
