param(
    [string]$Root = "E:\ensayos_genotipoXambiente",
    [int]$PollSeconds = 60,
    [int]$MissingFetcherGracePolls = 15
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Root
$Python = Join-Path $Root ".audit-venv\Scripts\python.exe"
$ReferenceRoot = Join-Path $Root "environment\v2\phase6a_cds_era5_land_bias_reference_v1"
$ReceiptRoot = Join-Path $ReferenceRoot "requests"
$Expected = 907

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python stage failed: $($Arguments -join ' ')"
    }
}

function Get-Sha256Lower {
    param([string]$Path)
    $Stream = [System.IO.File]::OpenRead($Path)
    try {
        $Hasher = [System.Security.Cryptography.SHA256]::Create()
        try {
            return ([System.BitConverter]::ToString($Hasher.ComputeHash($Stream))).Replace("-", "").ToLowerInvariant()
        } finally {
            $Hasher.Dispose()
        }
    } finally {
        $Stream.Dispose()
    }
}

Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WAIT CDS ERA5-Land 1981-2010 reference"
$MissingFetcherPolls = 0
while ($true) {
    $Completed = @(
        Get-ChildItem -LiteralPath $ReceiptRoot -Filter *.json -Recurse -File -ErrorAction SilentlyContinue
    ).Count
    Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] CDS_REFERENCE completed=$Completed expected=$Expected"
    if ($Completed -eq $Expected) {
        break
    }
    if ($Completed -gt $Expected) {
        throw "Unexpected extra CDS reference receipts: $Completed"
    }
    $Fetcher = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -like "*fetch_phase6a_cds_bias_reference*" -or
        $_.CommandLine -like "*run_phase6a_cds_bias_reference.ps1*"
    }
    if (-not $Fetcher) {
        $MissingFetcherPolls += 1
        Write-Warning (
            "CDS reference fetch is absent during a permitted restart window: " +
            "completed=$Completed grace_poll=$MissingFetcherPolls/$MissingFetcherGracePolls"
        )
        if ($MissingFetcherPolls -gt $MissingFetcherGracePolls) {
            throw (
                "CDS reference fetch remained absent beyond the restart grace window: " +
                "completed=$Completed grace_polls=$MissingFetcherGracePolls"
            )
        }
    } else {
        $MissingFetcherPolls = 0
    }
    Start-Sleep -Seconds $PollSeconds
}

$NormalizationProvenance = Join-Path $Root (
    "audit\v2\phase6a_daily_normalization_v1\cds_bias_reference\" +
    "cds_bias_reference_daily_normalization_provenance.json"
)
$ReferenceCube = Join-Path $Root (
    "environment\v2\e_projection_core_v1_historical_daily\cds_bias_reference\" +
    "cds_era5_land_1981_2010_daily_reference.nc"
)
$NormalizationReusable = $false
if ((Test-Path -LiteralPath $NormalizationProvenance) -and (Test-Path -LiteralPath $ReferenceCube)) {
    $Normalization = Get-Content -LiteralPath $NormalizationProvenance -Raw | ConvertFrom-Json
    $CubeHash = Get-Sha256Lower -Path $ReferenceCube
    $NormalizationReusable = (
        $Normalization.status -eq "PASS" -and
        $Normalization.site_count -eq 907 -and
        $Normalization.reference_cube_time_count -eq 10957 -and
        $Normalization.reference_cube_sha256 -eq $CubeHash
    )
}
if ($NormalizationReusable) {
    Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] SKIP checksum-certified authoritative reference normalization"
} else {
    Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] NORMALIZE authoritative reference"
    Invoke-CheckedPython @(
        "-m", "server_training_pipeline.normalize_phase6a_cds_bias_reference",
        "--root", ".", "--workers", "4"
    )
}

Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] FIT frozen historical-only bias parameters"
Invoke-CheckedPython @(
    "-m", "server_training_pipeline.fit_phase6a_historical_bias_adjustment",
    "--root", "."
)

Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] BUILD bias-adjusted historical CMIP6 backcast"
Invoke-CheckedPython @(
    "-m", "server_training_pipeline.build_phase6a_bias_adjusted_cmip6_backcast",
    "--root", "."
)

Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] CERTIFY feature parity and historical transfer"
Invoke-CheckedPython @(
    "-m", "server_training_pipeline.certify_phase6a_historical_transfer",
    "--root", "."
)

Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] CERTIFY E_PROJECTION_CORE_V1 readiness"
Invoke-CheckedPython @(
    "-m", "server_training_pipeline.certify_e_projection_core_v1_readiness",
    "--root", "."
)

Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] FREEZE E_PROJECTION_CORE_V1"
Invoke-CheckedPython @(
    "-m", "server_training_pipeline.freeze_e_projection_core_v1",
    "--root", "."
)

Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE E_PROJECTION_CORE_V1 historical transfer release"
Write-Output "Future covariate matrices and predictions remain ungenerated."
