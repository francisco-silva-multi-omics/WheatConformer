param(
    [string]$Root = "E:\ensayos_genotipoXambiente",
    [string]$Python = "E:\ensayos_genotipoXambiente\.audit-venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
Push-Location $Root
try {
    & $Python -m scripts.v2.fetch_stage1_v2_phase6_phenology_horizon_extension status --root $Root
    $latest = Join-Path $Root "audit\v2\stage1_v2_phase6_phenology_readiness_v1\latest_fetch_log_dir.txt"
    if (Test-Path $latest) {
        $logDir = (Get-Content $latest -Raw).Trim()
        Write-Host "Latest logs: $logDir"
        Get-ChildItem $logDir -Filter "*.stdout.log" | ForEach-Object {
            Write-Host "--- $($_.Name) ---"
            Get-Content $_.FullName -Tail 8
        }
        Get-ChildItem $logDir -Filter "*.stderr.log" | Where-Object Length -gt 0 | ForEach-Object {
            Write-Host "--- $($_.Name) errors ---"
            Get-Content $_.FullName -Tail 8
        }
    }
} finally {
    Pop-Location
}
