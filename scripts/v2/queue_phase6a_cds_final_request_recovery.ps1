$ErrorActionPreference = "Stop"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeRoot = (Resolve-Path (Join-Path $ScriptDirectory "..\..")).Path
$PythonCommand = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$PythonExecutable = (Get-Command $PythonCommand -ErrorAction Stop).Source
$RequestId = "3490488fff3a082fc82369213de57711ebec8164ca8c0c607266a1796c4487e9"

Set-Location $CodeRoot
$env:PYTHONPATH = if ($env:PYTHONPATH) {
    "$CodeRoot$([IO.Path]::PathSeparator)$env:PYTHONPATH"
} else {
    $CodeRoot
}
$env:PYTHONSAFEPATH = "1"

Write-Output "[$(Get-Date -Format o)] START partitioned recovery for final CDS request"
& $PythonExecutable -m server_training_pipeline.recover_phase6a_cds_partitioned_request `
    --root $CodeRoot `
    --request-id $RequestId
if ($LASTEXITCODE -ne 0) {
    throw "Partitioned CDS request recovery failed"
}

Write-Output "[$(Get-Date -Format o)] REINDEX complete CDS raw archive"
& $PythonExecutable -m server_training_pipeline.phase6a_environment_source_recovery `
    fetch-cds-era5-land `
    --root $CodeRoot `
    --contract-dir (Join-Path $CodeRoot "audit\v2\phase6a_environment_source_contract_v10") `
    --cache-dir (Join-Path $CodeRoot "environment\v2\phase6a_cds_era5_land_daily_full_v1") `
    --limit 0
if ($LASTEXITCODE -ne 0) {
    throw "CDS raw archive reindex failed"
}
Write-Output "[$(Get-Date -Format o)] DONE complete CDS raw archive"
