param(
    [int]$MaximumResolutionAttempts = 3,
    [int]$Workers = 2
)

$ErrorActionPreference = "Stop"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeRoot = (Resolve-Path (Join-Path $ScriptDirectory "..\..")).Path
$PythonCommand = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$PythonExecutable = (Get-Command $PythonCommand -ErrorAction Stop).Source

Set-Location $CodeRoot
$env:PYTHONPATH = if ($env:PYTHONPATH) {
    "$CodeRoot$([IO.Path]::PathSeparator)$env:PYTHONPATH"
} else {
    $CodeRoot
}
$env:PYTHONSAFEPATH = "1"

function Get-ResolutionWorkers {
    return @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(\.exe)?$' -and
        $_.CommandLine -like '*resolve_phase6a_soilgrids_missing*resolve*'
    })
}

for ($Attempt = 1; $Attempt -le $MaximumResolutionAttempts; $Attempt++) {
    $Active = @(Get-ResolutionWorkers)
    if (@($Active).Count -gt 0) {
        Write-Host "[$(Get-Date -Format o)] WAIT attempt=$Attempt active=$($Active.ProcessId -join ',')"
        while ($true) {
            $Remaining = @(Get-ResolutionWorkers)
            if (@($Remaining).Count -eq 0) { break }
            Start-Sleep -Seconds 30
        }
    } elseif ($Attempt -eq 1) {
        Write-Host "[$(Get-Date -Format o)] No active resolver found; checking current archive"
    }

    $ProvenancePath = Join-Path $CodeRoot "environment\v2\phase6a_soilgrids_missing_resolution_v1\soilgrids_missing_resolution_provenance.json"
    $Complete = $false
    if (Test-Path -LiteralPath $ProvenancePath -PathType Leaf) {
        $Provenance = Get-Content -LiteralPath $ProvenancePath -Raw | ConvertFrom-Json
        $Complete = $Provenance.run_status -eq "COMPLETE"
    }
    if ($Complete) {
        & $PythonExecutable -m server_training_pipeline.certify_phase6a_soilgrids_missing_resolution `
            --root $CodeRoot
        if ($LASTEXITCODE -ne 0) {
            throw "SoilGrids missing-resolution certification failed"
        }
        Write-Host "[$(Get-Date -Format o)] PASS SoilGrids missing-resolution certification"
        exit 0
    }

    if ($Attempt -lt $MaximumResolutionAttempts) {
        Write-Host "[$(Get-Date -Format o)] RESUME unresolved/retryable sites"
        & $PythonExecutable -m server_training_pipeline.resolve_phase6a_soilgrids_missing `
            resolve `
            --root $CodeRoot `
            --limit 0 `
            --workers $Workers `
            --timeout 120 `
            --retries 5
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Resolution attempt $Attempt exited nonzero; a bounded retry remains"
        }
    }
}

throw "SoilGrids missing-resolution archive remains incomplete after $MaximumResolutionAttempts attempts"
