param(
    [string]$PgHost = "localhost",
    [int]$PgPort = 5432,
    [string]$PgUser = "postgres",
    [string]$PgPassword = "postgres"
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$sql = Join-Path $root "migrations\001_layer1_schemas.sql"
$psql = if (Test-Path "C:\Program Files\PostgreSQL\16\bin\psql.exe") {
    "C:\Program Files\PostgreSQL\16\bin\psql.exe"
} else { "psql" }
$env:PGPASSWORD = $PgPassword
& $psql -h $PgHost -p $PgPort -U $PgUser -d icu_decision -f $sql
Write-Host "Applied migrations to icu_decision"
