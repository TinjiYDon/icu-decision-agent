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
Write-Host "OK. Ensure configs/data.yaml has source: mimic"
Write-Host "docker compose 默认映射端口 5433；训练前可设:"
Write-Host '  $env:DATABASE_URL = "postgresql+psycopg://icu_dev:icu_dev@localhost:5433/icu_decision"'
Write-Host "Connection: icu_dev/icu_dev @ ${PgHost}:${PgPort}/$db（restore 后若无权限，用 postgres GRANT USAGE/SELECT）"
