# Export Layer1 database icu_decision to dumps/ (do not commit dumps to GitHub)
# Usage: .\export_layer1.ps1 [-SchemasOnly] [-MimicSource mimic]
param(
    [switch]$SchemasOnly,
    [string]$OutDir = "",
    [string]$PgHost = "localhost",
    [int]$PgPort = 5432,
    [string]$PgUser = "postgres",
    [string]$PgPassword = "postgres",
    [string]$MimicSource = "mimic",
    [string]$Phase = "P0"
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if (-not $OutDir) { $OutDir = Join-Path $root "dumps" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$DbName = "icu_decision"
$Schemas = @("staging", "feat", "label", "model", "app")

$pgDump = "pg_dump"
$psql = "psql"
if (Test-Path "C:\Program Files\PostgreSQL\16\bin\pg_dump.exe") {
    $pgDump = "C:\Program Files\PostgreSQL\16\bin\pg_dump.exe"
    $psql = "C:\Program Files\PostgreSQL\16\bin\psql.exe"
}

$env:PGPASSWORD = $PgPassword
$dateTag = Get-Date -Format "yyyyMMdd"
$version = Get-Date -Format "yyyyMMdd_HHmm"

$stayCount = 0
$n = & $psql -h $PgHost -p $PgPort -U $PgUser -d $DbName -t -A -c "SELECT COUNT(*) FROM staging.icustays;" 2>$null
if ($LASTEXITCODE -eq 0 -and $n) { $stayCount = [int]$n.Trim() }

$kind = if ($SchemasOnly) { "etl" } else { "full" }
if ($stayCount -le 0 -and $SchemasOnly) { $kind = "schema-only" }
$out = Join-Path $OutDir "${DbName}_${Phase}-${kind}_${MimicSource}_${stayCount}stays_${dateTag}.dump"

$pgArgs = @("-h", $PgHost, "-p", "$PgPort", "-U", $PgUser, "-Fc", "-f", $out)
if ($SchemasOnly) {
    foreach ($s in $Schemas) { $pgArgs += @("-n", $s) }
}
$pgArgs += $DbName

Write-Host "Exporting $DbName -> $out"
& $pgDump @pgArgs
if ($LASTEXITCODE -ne 0) { throw "pg_dump failed for $DbName (exit $LASTEXITCODE)" }

$meta = @{
    version      = $version
    database     = $DbName
    phase        = $Phase
    mimic_source = $MimicSource
    stay_count   = $stayCount
    schemas_only = [bool]$SchemasOnly
    exported_at  = (Get-Date).ToString("o")
    file         = $out
}
$metaPath = Join-Path $OutDir "DATA_VERSION_${version}.json"
$meta | ConvertTo-Json -Depth 4 | Set-Content -Path $metaPath -Encoding UTF8

Write-Host ""
Write-Host "Done. Share dump offline if needed (not via GitHub):"
Write-Host "  $out"
Write-Host "  $metaPath"
