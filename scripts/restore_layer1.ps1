# Restore Layer1 dump into icu_decision
# Usage: .\restore_layer1.ps1 -DumpFile .\dumps\icu_decision_P0-etl_*.dump
param(
    [Parameter(Mandatory = $true)]
    [string]$DumpFile,
    [string]$PgHost = "localhost",
    [int]$PgPort = 5432,
    [string]$PgUser = "postgres",
    [string]$PgPassword = "postgres"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $DumpFile)) { throw "Dump not found: $DumpFile" }

$db = "icu_decision"
$pgRestore = "pg_restore"
if (Test-Path "C:\Program Files\PostgreSQL\16\bin\pg_restore.exe") {
    $pgRestore = "C:\Program Files\PostgreSQL\16\bin\pg_restore.exe"
}

$env:PGPASSWORD = $PgPassword
Write-Host "Restoring $DumpFile -> $db on ${PgHost}:${PgPort}"
# --no-acl：避免 docker 非超级用户角色上 ALTER DEFAULT PRIVILEGES 失败
& $pgRestore -h $PgHost -p $PgPort -U $PgUser -d $db --clean --if-exists --no-owner --no-acl $DumpFile
if ($LASTEXITCODE -ne 0) { throw "pg_restore failed (exit $LASTEXITCODE)" }

$psql = "psql"
if (Test-Path "C:\Program Files\PostgreSQL\16\bin\psql.exe") {
    $psql = "C:\Program Files\PostgreSQL\16\bin\psql.exe"
}
# App connects as icu_dev; full dump restore often drops grants
& $psql -h $PgHost -p $PgPort -U $PgUser -d $db -v ON_ERROR_STOP=1 -c @"
GRANT USAGE ON SCHEMA staging, feat, label, model, app, mock TO icu_dev;
GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA staging, feat, label, model, app, mock TO icu_dev;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA staging, feat, label, model, app, mock TO icu_dev;
"@
if ($LASTEXITCODE -ne 0) { throw "GRANT to icu_dev failed (exit $LASTEXITCODE)" }

Write-Host "OK. Ensure configs/data.yaml has source: mimic"
Write-Host "docker compose default port 5433; for Docker Layer1 set:"
Write-Host '  $env:DATABASE_URL = "postgresql+psycopg://icu_dev:icu_dev@localhost:5433/icu_decision"'
Write-Host "Connection: icu_dev/icu_dev @ ${PgHost}:${PgPort}/$db"
Write-Host "S2 accept: SELECT COUNT(*) FROM feat.sample_matrix;  -- expect ~472290"
