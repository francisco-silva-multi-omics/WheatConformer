param(
    [int]$Limit = 0,
    [string]$SourceId = ""
)

$ErrorActionPreference = "Stop"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeRoot = (Resolve-Path (Join-Path $ScriptDirectory "..\..")).Path
$PythonExecutable = Join-Path $CodeRoot ".audit-venv\Scripts\python.exe"
$AuditDirectory = Join-Path $CodeRoot "audit\v2\phase6a_cmip6_member_resolved_fetch_v1"
$CacheDirectory = Join-Path $CodeRoot "environment\v2\phase6a_cmip6_member_resolved_daily_v1"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDirectory = Join-Path $CodeRoot "logs\phase6a_cmip6_member_fetch_$Timestamp"

if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "Missing isolated Python environment: $PythonExecutable"
}

Set-Location $CodeRoot
$env:PYTHONPATH = if ($env:PYTHONPATH) {
    "$CodeRoot$([IO.Path]::PathSeparator)$env:PYTHONPATH"
} else {
    $CodeRoot
}
$env:PYTHONSAFEPATH = "1"

& $PythonExecutable -c "import xarray, gcsfs, cftime, h5netcdf, pydap"
if ($LASTEXITCODE -ne 0) {
    throw "Install dependencies with: $PythonExecutable -m pip install -r scripts/v2/phase6a_cmip6_member_fetch_requirements.txt"
}

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(\.exe)?$' -and
    $_.CommandLine -like '*fetch_cmip6_member_resolved*fetch*'
}
if ($existing) {
    $ids = ($existing | Select-Object -ExpandProperty ProcessId) -join ","
    throw "A member-resolved CMIP6 fetch is already running: $ids"
}

& $PythonExecutable -m server_training_pipeline.fetch_cmip6_member_resolved `
    prepare --root $CodeRoot --audit-dir $AuditDirectory
if ($LASTEXITCODE -ne 0) {
    throw "CMIP6 transport preparation failed"
}

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$stdout = Join-Path $LogDirectory "cmip6.stdout.log"
$stderr = Join-Path $LogDirectory "cmip6.stderr.log"
$arguments = @(
    "-m", "server_training_pipeline.fetch_cmip6_member_resolved",
    "fetch",
    "--root", $CodeRoot,
    "--audit-dir", $AuditDirectory,
    "--cache-dir", $CacheDirectory,
    "--limit", $Limit.ToString()
)
if ($SourceId) {
    $arguments += @("--source-id", $SourceId)
}

$process = Start-Process `
    -FilePath $PythonExecutable `
    -ArgumentList $arguments `
    -WorkingDirectory $CodeRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

$manifest = [pscustomobject]@{
    source = "member_resolved_cmip6"
    process_id = $process.Id
    started_at = (Get-Date).ToString("o")
    limit = $Limit
    source_id = $SourceId
    stdout_log = $stdout
    stderr_log = $stderr
    audit_directory = $AuditDirectory
    cache_directory = $CacheDirectory
}
$manifest | Export-Csv -LiteralPath (Join-Path $LogDirectory "fetch_process.tsv") -Delimiter "`t" -NoTypeInformation

Write-Host "Started exact-member CMIP6 historical + SSP site fetch"
$manifest | Format-List
Write-Host "This process does not use CDS and does not interfere with the active ERA5-Land request"
Write-Host "Future covariate matrices and predictions remain disabled"
Write-Host "Keep Windows awake and keep the E: drive connected"
