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
& $pgRestore -h $PgHost -p $PgPort -U $PgUser -d $db --clean --if-exists --no-owner --role=icu_dev $DumpFile
Write-Host "OK. Ensure configs/data.yaml has source: mimic"
Write-Host "Connection: icu_dev/icu_dev @ ${PgHost}:${PgPort}/$db"
