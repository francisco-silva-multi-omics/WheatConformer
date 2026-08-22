param(
    [int]$MaximumRetryPasses = 5
)

$ErrorActionPreference = "Stop"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeRoot = (Resolve-Path (Join-Path $ScriptDirectory "..\..")).Path
$PythonExecutable = Join-Path $CodeRoot ".audit-venv\Scripts\python.exe"
$CacheDirectory = Join-Path $CodeRoot "environment\v2\phase6a_cmip6_member_resolved_daily_v1"

Set-Location $CodeRoot
$env:PYTHONPATH = if ($env:PYTHONPATH) {
    "$CodeRoot$([IO.Path]::PathSeparator)$env:PYTHONPATH"
} else {
    $CodeRoot
}
$env:PYTHONSAFEPATH = "1"

function Get-ExistingCmipWorkers {
    return @(Get-CimInstance Win32_Process | Where-Object {
        ($_.Name -match '^python(\.exe)?$' -and $_.CommandLine -like '*fetch_cmip6_esgf_http*') -or
        ($_.Name -match '^powershell(\.exe)?$' -and $_.CommandLine -like '*-File*queue_phase6a_cmip6_complete_fetch.ps1*')
    })
}

function Get-CompletedAssetCount {
    $RequestDirectory = Join-Path $CacheDirectory "requests"
    if (-not (Test-Path -LiteralPath $RequestDirectory)) { return 0 }
    return @(Get-ChildItem -LiteralPath $RequestDirectory -Recurse -Filter *.json -File).Count
}

$Active = @(Get-ExistingCmipWorkers)
if (@($Active).Count -gt 0) {
    Write-Output "[$(Get-Date -Format o)] WAIT active CMIP6 transport PIDs=$($Active.ProcessId -join ',')"
    while ($true) {
        $Remaining = @(Get-ExistingCmipWorkers)
        if (@($Remaining).Count -eq 0) { break }
        Start-Sleep -Seconds 60
    }
}

$Previous = Get-CompletedAssetCount
for ($Pass = 1; $Pass -le $MaximumRetryPasses; $Pass++) {
    if ($Previous -ge 455) { break }
    Write-Output "[$(Get-Date -Format o)] RETRY pass=$Pass completed_before=$Previous"
    & $PythonExecutable -m server_training_pipeline.fetch_cmip6_esgf_http `
        --root $CodeRoot `
        --limit 0 `
        --retries 5 `
        --timeout 600
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "ESGF retry pass $Pass exited nonzero"
    }
    $Current = Get-CompletedAssetCount
    Write-Output "[$(Get-Date -Format o)] RETRY pass=$Pass completed_after=$Current"
    if ($Current -ge 455) { break }
    if ($Current -eq $Previous -and $Pass -ge 2) {
        Write-Warning "No progress in retry pass $Pass; unresolved assets require replica remediation"
        break
    }
    $Previous = $Current
}

& $PythonExecutable -m server_training_pipeline.fetch_cmip6_member_resolved `
    status --root $CodeRoot
if ($LASTEXITCODE -ne 0) {
    throw "Could not report final CMIP6 transport status"
}
