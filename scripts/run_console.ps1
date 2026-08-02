# Start ICU decision Plotly console (venv only; clear polluted env)
param(
    [int]$Port = 8501
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

# Do not inherit zhixue / other project DATABASE_URL
Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue
Remove-Item Env:SQLALCHEMY_DATABASE_URI -ErrorAction SilentlyContinue

$env:PYTHONPATH = $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Missing venv: $py" }

# Free port if leftover listeners
Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "Stopping PID $($_.OwningProcess) on port $Port"
    Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 1

Write-Host "Starting decision console at http://localhost:$Port"
Write-Host "Using: $py"
& $py -m streamlit run (Join-Path $root "presentation\streamlit_app.py") --server.port $Port --server.headless true
