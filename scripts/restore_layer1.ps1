# Layer 1 恢复（队友执行，dump 从网盘下载到 project-code/dumps/）
# 用法：
#   docker compose up -d
#   .\restore_layer1.ps1 -Target decision -DumpFile .\dumps\icu_decision_layer1_schemas_xxx.dump
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("decision", "scheduling")]
    [string]$Target,
    [Parameter(Mandatory = $true)]
    [string]$DumpFile,
    [string]$PgHost = "localhost",
    [int]$PgPort = 5432,
    [string]$PgUser = "postgres",
    [string]$PgPassword = "postgres"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $DumpFile)) { throw "Dump not found: $DumpFile" }

$db = if ($Target -eq "decision") { "icu_decision" } else { "icu_scheduling" }

$pgRestore = "pg_restore"
if (Test-Path "C:\Program Files\PostgreSQL\16\bin\pg_restore.exe") {
    $pgRestore = "C:\Program Files\PostgreSQL\16\bin\pg_restore.exe"
}

$env:PGPASSWORD = $PgPassword
Write-Host "Restoring $DumpFile -> $db on ${PgHost}:${PgPort}"
& $pgRestore -h $PgHost -p $PgPort -U $PgUser -d $db --clean --if-exists --no-owner --role=icu_dev $DumpFile
Write-Host "OK. Set configs/data.yaml source to mimic (or keep mock for UI-only)."
Write-Host "Connection: icu_dev/icu_dev @ ${PgHost}:${PgPort}/$db"
