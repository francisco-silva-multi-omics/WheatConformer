$ErrorActionPreference = "Stop"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeRoot = (Resolve-Path (Join-Path $ScriptDirectory "..\..")).Path
$PythonExecutable = Join-Path $CodeRoot ".audit-venv\Scripts\python.exe"

Set-Location $CodeRoot
$env:PYTHONPATH = if ($env:PYTHONPATH) {
    "$CodeRoot$([IO.Path]::PathSeparator)$env:PYTHONPATH"
} else {
    $CodeRoot
}
$env:PYTHONSAFEPATH = "1"

Write-Output "[$(Get-Date -Format o)] WAIT for the active exact-Zarr CMIP6 worker"
while ($true) {
    $active = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(\.exe)?$' -and
        $_.CommandLine -like '*fetch_cmip6_member_resolved*fetch*'
    }
    if (-not $active) { break }
    Start-Sleep -Seconds 60
}

Write-Output "[$(Get-Date -Format o)] RESUME exact-Zarr queue to close any interrupted assets"
& $PythonExecutable -m server_training_pipeline.fetch_cmip6_member_resolved fetch --root $CodeRoot --limit 0
if ($LASTEXITCODE -ne 0) {
    throw "Exact-Zarr CMIP6 fetch did not complete cleanly"
}

Write-Output "[$(Get-Date -Format o)] START exact ESGF HTTP fallback queue"
& $PythonExecutable -m server_training_pipeline.fetch_cmip6_esgf_http --root $CodeRoot --limit 0
if ($LASTEXITCODE -ne 0) {
    throw "Exact ESGF HTTP CMIP6 fallback did not complete cleanly"
}

Write-Output "[$(Get-Date -Format o)] DONE member-resolved CMIP6 transport queues"
& $PythonExecutable -m server_training_pipeline.fetch_cmip6_member_resolved status --root $CodeRoot
