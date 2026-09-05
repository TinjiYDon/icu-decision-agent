# Restore S2 dump and fix permissions for icu_dev user
# Usage: .\scripts\restore_s2.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = (Get-Location).Path
$DumpFile = Join-Path $ProjectRoot "dumps\icu_decision_S2-full_mimic_94458stays_20260802(1).dump"
$pgRestore = "C:\Program Files\PostgreSQL\16\bin\pg_restore.exe"
$psql = "C:\Program Files\PostgreSQL\16\bin\psql.exe"

if (-not (Test-Path $DumpFile)) {
    Write-Host "ERROR: Dump not found: $DumpFile" -ForegroundColor Red
    exit 1
}

Write-Host "=== Restoring S2 dump ===" -ForegroundColor Cyan
$env:PGPASSWORD = "lewis790919"
& $pgRestore -h localhost -U postgres -d icu_decision --clean --if-exists --no-owner --no-acl $DumpFile
if ($LASTEXITCODE -ne 0) { throw "pg_restore failed" }

Write-Host "=== Granting permissions to icu_dev ===" -ForegroundColor Cyan
& $psql -h localhost -U postgres -d icu_decision -v ON_ERROR_STOP=1 -c @"
GRANT ALL PRIVILEGES ON SCHEMA staging, feat, label, model, app, mock TO icu_dev;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA staging, feat, label, model, app, mock TO icu_dev;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA staging, feat, label, model, app, mock TO icu_dev;
ALTER DEFAULT PRIVILEGES IN SCHEMA staging, feat, label, model, app, mock GRANT ALL ON TABLES TO icu_dev;
ALTER DEFAULT PRIVILEGES IN SCHEMA staging, feat, label, model, app, mock GRANT ALL ON SEQUENCES TO icu_dev;
"@
if ($LASTEXITCODE -ne 0) { throw "GRANT failed" }

Write-Host "=== Verifying ===" -ForegroundColor Cyan
& $psql -h localhost -U icu_dev -d icu_decision -c "SELECT COUNT(*) as feat_rows FROM feat.sample_matrix;"
& $psql -h localhost -U icu_dev -d icu_decision -c "SELECT hour_index, COUNT(*) FROM feat.sample_matrix GROUP BY 1 ORDER BY 1;"

Write-Host "" -ForegroundColor Green
Write-Host "DONE. Streamlit: http://localhost:8501" -ForegroundColor Green
