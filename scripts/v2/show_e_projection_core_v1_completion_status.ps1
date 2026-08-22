param([string]$Root = "E:\ensayos_genotipoXambiente")

$ReceiptRoot = Join-Path $Root "environment\v2\phase6a_cds_era5_land_bias_reference_v1\requests"
$Completed = @(
    Get-ChildItem -LiteralPath $ReceiptRoot -Filter *.json -Recurse -File -ErrorAction SilentlyContinue
).Count
Write-Output "CDS reference: $Completed/907"

$Worker = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*queue_e_projection_core_v1_completion.ps1*"
}
if ($Worker) {
    Write-Output "E_PROJECTION_CORE_V1 completion supervisor is running"
    $Worker | Select-Object ProcessId, ParentProcessId, CreationDate, CommandLine | Format-List
} else {
    Write-Output "No E_PROJECTION_CORE_V1 completion supervisor is running"
}

$Logs = Get-ChildItem (Join-Path $Root "logs") -Directory -Filter "e_projection_core_v1_completion_*" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($Logs) {
    Write-Output "Latest completion logs: $($Logs.FullName)"
    $Out = Join-Path $Logs.FullName "stdout.log"
    $Err = Join-Path $Logs.FullName "stderr.log"
    if (Test-Path -LiteralPath $Out) {
        Get-Content -LiteralPath $Out -Tail 20
    }
    if ((Test-Path -LiteralPath $Err) -and (Get-Item -LiteralPath $Err).Length -gt 0) {
        Write-Output "Latest errors:"
        Get-Content -LiteralPath $Err -Tail 20
    }
}

$Readiness = Join-Path $Root "audit\v2\e_projection_core_v1_readiness\E_PROJECTION_CORE_V1_READINESS.json"
if (Test-Path -LiteralPath $Readiness) {
    Get-Content -LiteralPath $Readiness
}
