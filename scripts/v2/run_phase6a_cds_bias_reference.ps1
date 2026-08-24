param(
    [ValidateSet("Prepare", "Fetch", "Status")]
    [string]$Action = "Status",
    [int]$Limit = 0,
    [ValidateRange(1, 20)]
    [int]$Workers = 4
)

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
$command = $Action.ToLowerInvariant()
& $PythonExecutable -m server_training_pipeline.fetch_phase6a_cds_bias_reference `
    $command --root $CodeRoot --limit $Limit --workers $Workers
if ($LASTEXITCODE -ne 0) {
    throw "Phase-6A CDS bias-reference $Action failed"
}
