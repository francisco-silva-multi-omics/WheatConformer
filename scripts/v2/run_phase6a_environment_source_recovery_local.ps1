param(
    [int]$OpenMeteoLimit = 2500,
    [int]$CdsLimit = 1000,
    [int]$SoilGridsLimit = 250,
    [switch]$UseExistingCredentials
)

$ErrorActionPreference = "Stop"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeRoot = (Resolve-Path (Join-Path $ScriptDirectory "..\..")).Path
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$ContractDirectory = Join-Path $CodeRoot "audit\v2\phase6a_environment_source_contract_v10"
$OpenMeteoCache = Join-Path $CodeRoot "environment\v2\phase6a_openmeteo_era5_daily_full_v1"
$CdsCache = Join-Path $CodeRoot "environment\v2\phase6a_cds_era5_land_daily_full_v1"
$SoilGridsCache = Join-Path $CodeRoot "environment\v2\phase6a_soilgrids_water_full_v1"
$CredentialPath = Join-Path $HOME ".cdsapirc"

foreach ($Value in @($OpenMeteoLimit, $CdsLimit, $SoilGridsLimit)) {
    if ($Value -lt 0) {
        throw "Fetch limits must be nonnegative; zero means all remaining requests"
    }
}

Set-Location $CodeRoot
$env:PYTHONPATH = if ($env:PYTHONPATH) {
    "$CodeRoot$([IO.Path]::PathSeparator)$env:PYTHONPATH"
} else {
    $CodeRoot
}
$env:PYTHONSAFEPATH = "1"

& $Python -c "import cdsapi, rasterio"
if ($LASTEXITCODE -ne 0) {
    throw "Install dependencies with: $Python -m pip install -r scripts/v2/phase6a_environment_source_requirements.txt"
}

if ($UseExistingCredentials) {
    if (-not (Test-Path -LiteralPath $CredentialPath -PathType Leaf)) {
        throw "No CDS configuration exists at $CredentialPath; rerun without -UseExistingCredentials"
    }
    Write-Host "Using existing private CDS configuration: $CredentialPath"
} else {
    & $Python "scripts\v2\configure_cds_credentials.py"
    if ($LASTEXITCODE -ne 0) {
        throw "CDS credential configuration or verification failed"
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $ContractDirectory "environment_source_contract.json"))) {
    & $Python -m server_training_pipeline.phase6a_environment_source_recovery `
        build-contract `
        --root . `
        --out-dir $ContractDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "Phase-6A source contract construction failed"
    }
}

Write-Host "Fetching diagnostic Open-Meteo batch; limit=$OpenMeteoLimit"
& $Python -m server_training_pipeline.phase6a_environment_source_recovery `
    fetch-openmeteo `
    --root . `
    --contract-dir $ContractDirectory `
    --cache-dir $OpenMeteoCache `
    --limit $OpenMeteoLimit `
    --workers 2 `
    --timeout 120 `
    --retries 5
if ($LASTEXITCODE -ne 0) {
    throw "Open-Meteo batch failed"
}

Write-Host "Fetching authoritative CDS ERA5-Land batch; limit=$CdsLimit"
# The CDS fetcher is deliberately serial: one blocking retrieve call at a time.
& $Python -m server_training_pipeline.phase6a_environment_source_recovery `
    fetch-cds-era5-land `
    --root . `
    --contract-dir $ContractDirectory `
    --cache-dir $CdsCache `
    --limit $CdsLimit
if ($LASTEXITCODE -ne 0) {
    throw "CDS ERA5-Land batch failed"
}

Write-Host "Fetching SoilGrids batch; limit=$SoilGridsLimit"
& $Python -m server_training_pipeline.phase6a_environment_source_recovery `
    fetch-soilgrids `
    --root . `
    --contract-dir $ContractDirectory `
    --cache-dir $SoilGridsCache `
    --limit $SoilGridsLimit `
    --timeout 120 `
    --retries 5
if ($LASTEXITCODE -ne 0) {
    throw "SoilGrids batch failed"
}

Write-Host "DONE resumable Phase-6A source batch"
Write-Host "No phenotype, protected outcome, future covariate matrix, or prediction was read or generated"
Write-Host "Rerun with -UseExistingCredentials to continue from the content-addressed caches"
