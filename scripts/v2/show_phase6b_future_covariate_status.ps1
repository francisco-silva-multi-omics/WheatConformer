param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
)

$Processes = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*build_phase6b_member_resolved_future_covariates*" -or
    $_.CommandLine -like "*certify_phase6b_member_resolved_future_covariates*"
}

$Index = Join-Path $Root "audit\v2\e_projection_core_v1_future_covariates_v1\future_covariate_matrix_index.tsv"
$Receipts = Join-Path $Root "environment\v2\e_projection_core_v1_future_covariates_v1\receipts"
$Decision = Join-Path $Root "audit\v2\e_projection_core_v1_future_covariates_v1_release\FUTURE_COVARIATE_RELEASE_DECISION.json"

if ($Processes) {
    Write-Host "Phase 6B future-covariate worker is running"
    $Processes | Select-Object ProcessId, ParentProcessId, CreationDate, Name, CommandLine | Format-List
} else {
    Write-Host "No Phase 6B future-covariate worker is running"
}

if (Test-Path -LiteralPath $Receipts -PathType Container) {
    $ReceiptCount = @(Get-ChildItem -LiteralPath $Receipts -Filter "*.json" -File).Count
    Write-Host "Completed generation receipts: $ReceiptCount/52 groups ($($ReceiptCount * 2)/104 matrices)"
} elseif (Test-Path -LiteralPath $Index -PathType Leaf) {
    $Rows = Import-Csv -LiteralPath $Index -Delimiter "`t"
    Write-Host "Generated matrix index: $($Rows.Count)/104 matrices"
}
if (Test-Path -LiteralPath $Decision -PathType Leaf) {
    Get-Content -LiteralPath $Decision
}
