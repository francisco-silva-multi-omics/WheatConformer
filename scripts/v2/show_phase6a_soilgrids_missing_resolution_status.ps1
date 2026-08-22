$ErrorActionPreference = "Stop"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeRoot = (Resolve-Path (Join-Path $ScriptDirectory "..\..")).Path
$CacheDirectory = Join-Path $CodeRoot "environment\v2\phase6a_soilgrids_missing_resolution_v1"
$Provenance = Join-Path $CacheDirectory "soilgrids_missing_resolution_provenance.json"

if (Test-Path -LiteralPath $Provenance -PathType Leaf) {
    Get-Content -LiteralPath $Provenance -Raw
} else {
    $RequestDirectory = Join-Path $CacheDirectory "requests"
    $Counts = @{}
    if (Test-Path -LiteralPath $RequestDirectory) {
        foreach ($Metadata in Get-ChildItem -LiteralPath $RequestDirectory -Recurse -Filter *.json -File) {
            try {
                $Record = Get-Content -LiteralPath $Metadata.FullName -Raw | ConvertFrom-Json
                $Status = [string]$Record.status
                if (-not $Counts.ContainsKey($Status)) { $Counts[$Status] = 0 }
                $Counts[$Status] += 1
            } catch {
                Write-Warning "Could not read $($Metadata.FullName): $($_.Exception.Message)"
            }
        }
    }
    [pscustomobject]@{
        run_status = "IN_PROGRESS"
        resolved_sites = ($Counts.Values | Measure-Object -Sum).Sum
        total_sites = 212
        pending_sites = 212 - ($Counts.Values | Measure-Object -Sum).Sum
        status_counts = $Counts
        observations_excluded = 0
    } | ConvertTo-Json -Depth 4
}

$Processes = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(\.exe)?$' -and
    $_.CommandLine -like '*resolve_phase6a_soilgrids_missing*resolve*'
}
if ($Processes) {
    Write-Host "`nActive SoilGrids resolution worker:"
    $Processes | Select-Object ProcessId, CreationDate, CommandLine | Format-Table -Wrap
} else {
    Write-Host "`nNo SoilGrids missing-resolution worker is running"
}

$LatestLogs = Get-ChildItem (Join-Path $CodeRoot "logs") `
    -Directory -Filter "phase6a_soilgrids_missing_resolution_*" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($LatestLogs) {
    Write-Host "Latest logs: $($LatestLogs.FullName)"
    $Stdout = Join-Path $LatestLogs.FullName "soilgrids_resolution.stdout.log"
    $Stderr = Join-Path $LatestLogs.FullName "soilgrids_resolution.stderr.log"
    if (Test-Path -LiteralPath $Stdout) {
        Write-Host "`nLatest completed sites:"
        Get-Content -LiteralPath $Stdout -Tail 8
    }
    if ((Test-Path -LiteralPath $Stderr) -and (Get-Item $Stderr).Length -gt 0) {
        Write-Host "`nLatest errors:"
        Get-Content -LiteralPath $Stderr -Tail 12
    }
}
