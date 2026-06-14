param(
    [string]$PgHost = "localhost",
    [int]$PgPort = 5432,
    [string]$PgUser = "postgres",
    [string]$PgPassword = "postgres"
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$migrations = Join-Path $root "migrations"
$psql = if (Test-Path "C:\Program Files\PostgreSQL\16\bin\psql.exe") {
    "C:\Program Files\PostgreSQL\16\bin\psql.exe"
} else { "psql" }
$env:PGPASSWORD = $PgPassword
Get-ChildItem (Join-Path $migrations "*.sql") | Sort-Object Name | ForEach-Object {
    Write-Host "Applying $($_.Name) ..."
    & $psql -h $PgHost -p $PgPort -U $PgUser -d icu_decision -f $_.FullName
}
Write-Host "Applied all migrations to icu_decision"
