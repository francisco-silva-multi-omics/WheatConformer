param(
    [string]$Root = "E:\ensayos_genotipoXambiente",
    [string]$Python = "E:\ensayos_genotipoXambiente\.audit-venv\Scripts\python.exe",
    [int]$CdsWorkers = 4,
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $Root "logs\stage1_v2_phase6_phenology_horizon_$timestamp"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$commands = @(
    @{
        Name = "cds"
        Arguments = @(
            "-m", "scripts.v2.fetch_stage1_v2_phase6_phenology_horizon_extension",
            "fetch-cds", "--root", $Root, "--workers", "$CdsWorkers", "--limit", "$Limit"
        )
    }
)

foreach ($command in $commands) {
    $stdout = Join-Path $logDir "$($command.Name).stdout.log"
    $stderr = Join-Path $logDir "$($command.Name).stderr.log"
    $process = Start-Process -FilePath $Python `
        -ArgumentList $command.Arguments `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
    "$($command.Name)_pid=$($process.Id)" | Tee-Object -FilePath (Join-Path $logDir "pids.txt") -Append
}

$logDir | Set-Content -Encoding UTF8 (Join-Path $Root "audit\v2\stage1_v2_phase6_phenology_readiness_v1\latest_fetch_log_dir.txt")
Write-Host "Started reuse-first CDS phenology horizon fetch"
Write-Host "No new Open-Meteo fetch is required; the frozen 7,094-request cross-provider audit is reused"
Write-Host "Logs: $logDir"
