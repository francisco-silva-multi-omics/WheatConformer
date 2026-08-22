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

& $PythonExecutable -m server_training_pipeline.fetch_cmip6_member_resolved `
    status --root $CodeRoot

$processes = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(\.exe)?$' -and
    $_.CommandLine -like '*fetch_cmip6_member_resolved*fetch*'
}
if ($processes) {
    Write-Host "`nActive exact-Zarr CMIP6 fetch:"
    $processes | Select-Object ProcessId, CreationDate, CommandLine | Format-Table -Wrap
} else {
    Write-Host "`nNo exact-Zarr CMIP6 fetch process is running"
}

$fallbackProcesses = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(\.exe)?$' -and
    $_.CommandLine -like '*fetch_cmip6_esgf_http*'
}
if ($fallbackProcesses) {
    Write-Host "`nActive exact-ESGF fallback fetch:"
    $fallbackProcesses | Select-Object ProcessId, CreationDate, CommandLine | Format-Table -Wrap
} else {
    Write-Host "`nNo exact-ESGF fallback fetch process is running"
}

$latest = Get-ChildItem (Join-Path $CodeRoot "logs") `
    -Directory -Filter "phase6a_cmip6_member_fetch_*" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($latest) {
    Write-Host "Latest CMIP6 logs: $($latest.FullName)"
    $stdout = Join-Path $latest.FullName "cmip6.stdout.log"
    $stderr = Join-Path $latest.FullName "cmip6.stderr.log"
    if (Test-Path -LiteralPath $stdout) {
        Write-Host "`nLatest output:"
        Get-Content -LiteralPath $stdout -Tail 12
    }
    if ((Test-Path -LiteralPath $stderr) -and (Get-Item $stderr).Length -gt 0) {
        Write-Host "`nLatest errors:"
        Get-Content -LiteralPath $stderr -Tail 12
    }
}

$queue = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^powershell(\.exe)?$' -and
    ($_.CommandLine -like '*-File*queue_phase6a_cmip6_complete_fetch.ps1*' -or
     $_.CommandLine -like '*-File*queue_phase6a_cmip6_retry_closure.ps1*')
}
if ($queue) {
    Write-Host "`nQueued complete-transport supervisor:"
    $queue | Select-Object ProcessId, CreationDate, CommandLine | Format-Table -Wrap
}

$priorityLogPointer = Join-Path $CodeRoot "logs\phase6a_cmip6_transport_priority_latest.txt"
if (Test-Path -LiteralPath $priorityLogPointer) {
    $priorityLog = (Get-Content -LiteralPath $priorityLogPointer -Raw).Trim()
    if (Test-Path -LiteralPath $priorityLog) {
        Write-Host "Transport-priority logs: $priorityLog"
        $priorityStdout = Join-Path $priorityLog "stdout.log"
        $priorityStderr = Join-Path $priorityLog "stderr.log"
        if (Test-Path -LiteralPath $priorityStdout) {
            Get-Content -LiteralPath $priorityStdout -Tail 8
        }
        if ((Test-Path -LiteralPath $priorityStderr) -and
            (Get-Item -LiteralPath $priorityStderr).Length -gt 0) {
            Write-Host "`nLatest transport-priority errors:"
            Get-Content -LiteralPath $priorityStderr -Tail 8
        }
    }
}

$queueLog = Get-ChildItem (Join-Path $CodeRoot "logs") `
    -Directory -Filter "phase6a_cmip6_completion_queue_*" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($queueLog) {
    Write-Host "Completion-queue logs: $($queueLog.FullName)"
    $queueStdout = Join-Path $queueLog.FullName "queue.stdout.log"
    if (Test-Path -LiteralPath $queueStdout) {
        Get-Content -LiteralPath $queueStdout -Tail 5
    }
}
