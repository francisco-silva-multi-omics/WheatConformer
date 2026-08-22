param(
    [int]$Limit = 0,
    [int]$Workers = 2
)

$ErrorActionPreference = "Stop"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeRoot = (Resolve-Path (Join-Path $ScriptDirectory "..\..")).Path
$PythonCommand = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$PythonExecutable = (Get-Command $PythonCommand -ErrorAction Stop).Source
$AuditDirectory = Join-Path $CodeRoot "audit\v2\phase6a_soilgrids_missing_resolution_v1"
$CacheDirectory = Join-Path $CodeRoot "environment\v2\phase6a_soilgrids_missing_resolution_v1"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDirectory = Join-Path $CodeRoot "logs\phase6a_soilgrids_missing_resolution_$Timestamp"

if ($Limit -lt 0 -or $Workers -lt 1) {
    throw "Limit must be nonnegative and Workers must be positive"
}

Set-Location $CodeRoot
$env:PYTHONPATH = if ($env:PYTHONPATH) {
    "$CodeRoot$([IO.Path]::PathSeparator)$env:PYTHONPATH"
} else {
    $CodeRoot
}
$env:PYTHONSAFEPATH = "1"

& $PythonExecutable -c "import pandas, numpy, rasterio, pyarrow"
if ($LASTEXITCODE -ne 0) {
    throw "The selected Python lacks pandas, numpy, rasterio or pyarrow"
}

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(\.exe)?$' -and
    $_.CommandLine -like '*resolve_phase6a_soilgrids_missing*resolve*'
}
if ($existing) {
    $ids = ($existing | Select-Object -ExpandProperty ProcessId) -join ","
    throw "SoilGrids missing-resolution worker is already running: $ids"
}

if (-not (Test-Path -LiteralPath $AuditDirectory)) {
    & $PythonExecutable -m server_training_pipeline.resolve_phase6a_soilgrids_missing `
        freeze `
        --root $CodeRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Could not freeze the SoilGrids missing-resolution audit"
    }
}

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$Stdout = Join-Path $LogDirectory "soilgrids_resolution.stdout.log"
$Stderr = Join-Path $LogDirectory "soilgrids_resolution.stderr.log"
$Arguments = @(
    "-m", "server_training_pipeline.resolve_phase6a_soilgrids_missing",
    "resolve",
    "--root", $CodeRoot,
    "--limit", $Limit,
    "--workers", $Workers,
    "--timeout", 120,
    "--retries", 5
)
$Process = Start-Process `
    -FilePath $PythonExecutable `
    -ArgumentList $Arguments `
    -WorkingDirectory $CodeRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -PassThru

[pscustomobject]@{
    process_id = $Process.Id
    limit = $Limit
    workers = $Workers
    started_at = (Get-Date).ToString("o")
    stdout_log = $Stdout
    stderr_log = $Stderr
    audit_directory = $AuditDirectory
    cache_directory = $CacheDirectory
} | ConvertTo-Json -Depth 3

Write-Host "No observations will be excluded; unresolved sites remain explicitly masked"
