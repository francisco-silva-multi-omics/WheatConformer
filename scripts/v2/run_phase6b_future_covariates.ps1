param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [int]$Workers = 2
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $Root ".audit-venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Audit Python is missing: $Python"
}

function Invoke-Stage {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python stage failed: $($Arguments -join ' ')"
    }
}

Set-Location $Root
$Lock = Join-Path $Root "audit\v2\e_projection_core_v1_future_covariates_v1_freeze\future_covariate_generation_lock.json"
$ExtremeReference = Join-Path $Root "environment\v2\e_projection_core_v1_future_covariates_v1_reference\historical_daily_extreme_reference.nc"
$Decision = Join-Path $Root "audit\v2\e_projection_core_v1_future_covariates_v1_release\FUTURE_COVARIATE_RELEASE_DECISION.json"

if (-not (Test-Path -LiteralPath $Lock -PathType Leaf)) {
    Invoke-Stage @("-m", "server_training_pipeline.freeze_phase6b_future_covariates", "--root", ".")
}
if (-not (Test-Path -LiteralPath $ExtremeReference -PathType Leaf)) {
    Invoke-Stage @("-m", "server_training_pipeline.build_phase6b_daily_extreme_reference", "--root", ".")
}
Invoke-Stage @(
    "-m", "server_training_pipeline.build_phase6b_member_resolved_future_covariates",
    "--root", ".", "--workers", "$Workers"
)
if (-not (Test-Path -LiteralPath $Decision -PathType Leaf)) {
    Invoke-Stage @(
        "-m", "server_training_pipeline.certify_phase6b_member_resolved_future_covariates",
        "--root", "."
    )
}

Get-Content -LiteralPath $Decision
