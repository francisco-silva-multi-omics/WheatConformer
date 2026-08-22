param(
    [Parameter(Mandatory = $false)]
    [string]$Root = "."
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path $Root).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "[1/3] Freeze Stage-1 v2 split-bound historical projection-input contract"
& $Python -m server_training_pipeline.freeze_phase6a_split_bound_projection_inputs --root $Root
if ($LASTEXITCODE -ne 0) { throw "Projection-input freeze failed" }

Write-Host "[2/3] Build 150 training-only preprocessing and factorization states"
& $Python -m server_training_pipeline.build_phase6a_split_bound_projection_inputs --root $Root
if ($LASTEXITCODE -ne 0) { throw "Projection-input build failed" }

Write-Host "[3/3] Independently certify schema, masks, transforms, and leakage boundaries"
& $Python -m server_training_pipeline.certify_phase6a_split_bound_projection_inputs --root $Root
if ($LASTEXITCODE -ne 0) { throw "Projection-input certification failed" }

Write-Host "DONE split-bound historical E_PROJECTION_CORE_V1 inputs"
