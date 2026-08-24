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
& $PythonExecutable -m server_training_pipeline.fetch_phase6a_cds_bias_reference `
    status --root $CodeRoot

$workers = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*fetch_phase6a_cds_bias_reference*fetch*"
}
if ($workers) {
    Write-Output ""
    Write-Output "CDS bias-reference fetch is running"
    $workers | Select-Object ProcessId, ParentProcessId, Name | Format-Table -AutoSize
} else {
    Write-Output ""
    Write-Output "No CDS bias-reference fetch process is running"
}

$latest = Get-ChildItem -LiteralPath (Join-Path $CodeRoot "logs") -Directory |
    Where-Object { $_.Name -like "phase6a_cds_bias_reference_*" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($latest) {
    Write-Output "Latest logs: $($latest.FullName)"
    $stdout = Join-Path $latest.FullName "stdout.log"
    if (Test-Path -LiteralPath $stdout) {
        Write-Output ""
        Get-Content -LiteralPath $stdout -Tail 12
    }
}
