$ErrorActionPreference = "Stop"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeRoot = (Resolve-Path (Join-Path $ScriptDirectory "..\..")).Path

function Count-Files {
    param([string]$Path, [string]$Filter)
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    return (Get-ChildItem -LiteralPath $Path -Recurse -Filter $Filter -File).Count
}

$openCache = Join-Path $CodeRoot "environment\v2\phase6a_openmeteo_era5_daily_full_v1"
$cdsCache = Join-Path $CodeRoot "environment\v2\phase6a_cds_era5_land_daily_full_v1"
$soilCache = Join-Path $CodeRoot "environment\v2\phase6a_soilgrids_water_full_v1"
$soilResolutionCertification = Join-Path $CodeRoot "audit\v2\phase6a_soilgrids_missing_resolution_v1\soilgrids_missing_resolution_certification.json"

$soilNumeric = Count-Files (Join-Path $soilCache "values") "*.parquet"
$soilStructural = 0
if (Test-Path -LiteralPath (Join-Path $soilCache "requests")) {
    foreach ($metadata in Get-ChildItem (Join-Path $soilCache "requests") -Recurse -Filter *.json -File) {
        try {
            $record = Get-Content -LiteralPath $metadata.FullName -Raw | ConvertFrom-Json
            if ($record.status -eq "STRUCTURALLY_UNAVAILABLE_SOIL_CELL") {
                $soilStructural += 1
            }
        } catch {
            Write-Warning "Could not read $($metadata.FullName): $($_.Exception.Message)"
        }
    }
}

$summary = @(
    [pscustomobject]@{
        source = "Open-Meteo ERA5"
        resolved = Count-Files (Join-Path $openCache "requests") "*.json"
        total = 7094
        terminal_missing = 0
    }
    [pscustomobject]@{
        source = "CDS ERA5-Land"
        resolved = Count-Files (Join-Path $cdsCache "requests") "*.json"
        total = 7094
        terminal_missing = 0
    }
    [pscustomobject]@{
        source = "SoilGrids exact archive"
        resolved = $soilNumeric + $soilStructural
        total = 907
        terminal_missing = $soilStructural
    }
)
if (Test-Path -LiteralPath $soilResolutionCertification -PathType Leaf) {
    $soilResolution = Get-Content -LiteralPath $soilResolutionCertification -Raw | ConvertFrom-Json
    if ($soilResolution.status -eq "PASS") {
        $summary += [pscustomobject]@{
            source = "SoilGrids effective overlay"
            resolved = $soilNumeric + [int]$soilResolution.accepted_nearest_cell_sites
            total = 907
            terminal_missing = [int]$soilResolution.explicit_mask_sites
        }
    }
}
$summary | Format-Table -AutoSize

$processes = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(\.exe)?$' -and
    ($_.CommandLine -like '*phase6a_environment_source_recovery*fetch-*' -or
     $_.CommandLine -like '*recover_phase6a_cds_partitioned_request*')
}
if ($processes) {
    Write-Host "`nActive fetchers:"
    $processes | Select-Object ProcessId, CreationDate, CommandLine | Format-Table -Wrap
} else {
    Write-Host "`nNo Phase-6A fetch process is running"
}

$recoverySupervisors = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^powershell(\.exe)?$' -and
    $_.CommandLine -like '*-File*queue_phase6a_cds_final_request_recovery.ps1*'
}
if ($recoverySupervisors) {
    Write-Host "`nActive CDS final-request recovery supervisor:"
    $recoverySupervisors | Select-Object ProcessId, CreationDate, CommandLine | Format-Table -Wrap
}

$latestLogs = Get-ChildItem (Join-Path $CodeRoot "logs") `
    -Directory -Filter "phase6a_environment_sources_*" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($latestLogs) {
    Write-Host "Latest unattended logs: $($latestLogs.FullName)"
}
