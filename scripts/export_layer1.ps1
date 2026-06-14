# Layer 1 导出（数据负责人执行）
# 用法：
#   .\export_layer1.ps1 -Target all
#   .\export_layer1.ps1 -Target decision -SchemasOnly
param(
    [ValidateSet("decision", "scheduling", "all")]
    [string]$Target = "all",
    [switch]$SchemasOnly,
    [string]$OutDir = "",
    [string]$PgHost = "localhost",
    [int]$PgPort = 5432,
    [string]$PgUser = "postgres",
    [string]$PgPassword = "postgres"
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if (-not $OutDir) { $OutDir = Join-Path $root "dumps" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$pgDump = "pg_dump"
if (Test-Path "C:\Program Files\PostgreSQL\16\bin\pg_dump.exe") {
    $pgDump = "C:\Program Files\PostgreSQL\16\bin\pg_dump.exe"
}

$env:PGPASSWORD = $PgPassword
$version = Get-Date -Format "yyyyMMdd_HHmm"
$meta = @{
    version      = $version
    p0_layer0    = @("mimic_iv_demo", "mimic_iv")
    target       = $Target
    schemas_only = [bool]$SchemasOnly
    exported_at  = (Get-Date).ToString("o")
}

function Export-Db {
    param([string]$DbName, [string[]]$Schemas)
    $suffix = if ($SchemasOnly) { "_layer1_schemas" } else { "_layer1_full" }
    $out = Join-Path $OutDir "${DbName}${suffix}_${version}.dump"
    $pgArgs = @("-h", $PgHost, "-p", "$PgPort", "-U", $PgUser, "-Fc", "-f", $out)
    if ($SchemasOnly) {
        foreach ($s in $Schemas) {
            $pgArgs = $pgArgs + @("-n", $s)
        }
    }
    $pgArgs = $pgArgs + @($DbName)
    Write-Host "Exporting $DbName -> $out"
    & $pgDump @pgArgs
    if ($LASTEXITCODE -ne 0) { throw "pg_dump failed for $DbName (exit $LASTEXITCODE)" }
    return $out
}

$files = @()
if ($Target -in @("decision", "all")) {
    $files += Export-Db "icu_decision" @("staging", "feat", "label", "model", "app")
}
if ($Target -in @("scheduling", "all")) {
    $files += Export-Db "icu_scheduling" @("staging", "feat", "sim", "sched", "app")
}

$meta.files = $files
$metaPath = Join-Path $OutDir "DATA_VERSION_${version}.json"
$meta | ConvertTo-Json -Depth 4 | Set-Content -Path $metaPath -Encoding UTF8

Write-Host ""
Write-Host "Done. Upload to team cloud drive (do NOT commit to GitHub):"
Write-Host "  $metaPath"
foreach ($f in $files) { Write-Host "  $f" }
