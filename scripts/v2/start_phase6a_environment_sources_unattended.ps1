param(
    [switch]$UseExistingCredentials
)

$ErrorActionPreference = "Stop"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeRoot = (Resolve-Path (Join-Path $ScriptDirectory "..\..")).Path
$PythonCommand = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$PythonExecutable = (Get-Command $PythonCommand -ErrorAction Stop).Source
$ContractDirectory = Join-Path $CodeRoot "audit\v2\phase6a_environment_source_contract_v10"
$OpenMeteoCache = Join-Path $CodeRoot "environment\v2\phase6a_openmeteo_era5_daily_full_v1"
$CdsCache = Join-Path $CodeRoot "environment\v2\phase6a_cds_era5_land_daily_full_v1"
$SoilGridsCache = Join-Path $CodeRoot "environment\v2\phase6a_soilgrids_water_full_v1"
$CredentialPath = Join-Path $HOME ".cdsapirc"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDirectory = Join-Path $CodeRoot "logs\phase6a_environment_sources_$Timestamp"

Set-Location $CodeRoot
$env:PYTHONPATH = if ($env:PYTHONPATH) {
    "$CodeRoot$([IO.Path]::PathSeparator)$env:PYTHONPATH"
} else {
    $CodeRoot
}
$env:PYTHONSAFEPATH = "1"

if (-not (Test-Path -LiteralPath $CredentialPath -PathType Leaf)) {
    if ($UseExistingCredentials) {
        throw "No CDS configuration exists at $CredentialPath"
    }
    & $PythonExecutable "scripts\v2\configure_cds_credentials.py"
    if ($LASTEXITCODE -ne 0) {
        throw "CDS credential configuration or verification failed"
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $ContractDirectory "environment_source_contract.json"))) {
    throw "The frozen Phase-6A source contract is missing: $ContractDirectory"
}

& $PythonExecutable -c "import cdsapi, rasterio"
if ($LASTEXITCODE -ne 0) {
    throw "Install dependencies with: $PythonExecutable -m pip install -r scripts/v2/phase6a_environment_source_requirements.txt"
}

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(\.exe)?$' -and
    $_.CommandLine -like '*phase6a_environment_source_recovery*fetch-*'
}
if ($existing) {
    $ids = ($existing | Select-Object -ExpandProperty ProcessId) -join ","
    throw "Phase-6A fetch processes are already running: $ids"
}

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null

function Start-SourceFetch {
    param(
        [string]$Source,
        [string[]]$Arguments
    )

    $stdout = Join-Path $LogDirectory "$Source.stdout.log"
    $stderr = Join-Path $LogDirectory "$Source.stderr.log"
    $process = Start-Process `
        -FilePath $PythonExecutable `
        -ArgumentList $Arguments `
        -WorkingDirectory $CodeRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru

    [pscustomobject]@{
        source = $Source
        process_id = $process.Id
        started_at = (Get-Date).ToString("o")
        stdout_log = $stdout
        stderr_log = $stderr
        request_limit = 0
    }
}

$common = @(
    "-m",
    "server_training_pipeline.phase6a_environment_source_recovery"
)

$jobs = @()
$jobs += Start-SourceFetch "openmeteo" ($common + @(
    "fetch-openmeteo",
    "--root", ".",
    "--contract-dir", $ContractDirectory,
    "--cache-dir", $OpenMeteoCache,
    "--limit", "0",
    "--workers", "2",
    "--timeout", "120",
    "--retries", "5"
))
$jobs += Start-SourceFetch "cds_era5_land" ($common + @(
    "fetch-cds-era5-land",
    "--root", ".",
    "--contract-dir", $ContractDirectory,
    "--cache-dir", $CdsCache,
    "--limit", "0"
))
$jobs += Start-SourceFetch "soilgrids" ($common + @(
    "fetch-soilgrids",
    "--root", ".",
    "--contract-dir", $ContractDirectory,
    "--cache-dir", $SoilGridsCache,
    "--limit", "0",
    "--timeout", "120",
    "--retries", "5"
))

$manifest = Join-Path $LogDirectory "fetch_processes.tsv"
$jobs | Export-Csv -LiteralPath $manifest -Delimiter "`t" -NoTypeInformation

Write-Host "Started independent resumable Phase-6A historical fetchers"
$jobs | Format-Table source, process_id, request_limit, stdout_log, stderr_log -AutoSize
Write-Host "Process manifest: $manifest"
Write-Host "CDS concurrency remains exactly one request at a time"
Write-Host "CMIP6 was not started because ensemble identities are not preregistered"
Write-Host "Keep Windows awake and keep the E: drive connected"
