$ErrorActionPreference = "Stop"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeRoot = (Resolve-Path (Join-Path $ScriptDirectory "..\..")).Path
$PythonExecutable = Join-Path $CodeRoot ".audit-venv\Scripts\python.exe"
$LogDirectory = Join-Path $CodeRoot (
    "logs\phase6a_cmip6_earthdatahub_gap_recovery_" + (Get-Date -Format "yyyyMMdd_HHmmss")
)

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
Set-Content -LiteralPath (
    Join-Path $CodeRoot "logs\phase6a_cmip6_earthdatahub_gap_recovery_latest.txt"
) -Value $LogDirectory -Encoding ASCII

$SecureToken = Read-Host "Paste the Earth Data Hub API key (input hidden)" -AsSecureString
$TokenPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureToken)
try {
    $PlainToken = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($TokenPointer)
    if ([string]::IsNullOrWhiteSpace($PlainToken)) {
        throw "Earth Data Hub API key was empty"
    }
    $env:EARTHDATAHUB_API_KEY = $PlainToken
    $env:PYTHONPATH = if ($env:PYTHONPATH) {
        "$CodeRoot$([IO.Path]::PathSeparator)$env:PYTHONPATH"
    } else {
        $CodeRoot
    }
    $env:PYTHONSAFEPATH = "1"
    Set-Location $CodeRoot
    Write-Output "[$(Get-Date -Format o)] START Earth Data Hub concordance-gated CMIP6 gap recovery"
    & $PythonExecutable -m server_training_pipeline.recover_cmip6_earthdatahub_gap `
        --root $CodeRoot 2>&1 | Tee-Object -FilePath (Join-Path $LogDirectory "recovery.log")
    if ($LASTEXITCODE -ne 0) {
        throw "Earth Data Hub gap recovery failed; inspect $LogDirectory\recovery.log"
    }
} finally {
    Remove-Item Env:\EARTHDATAHUB_API_KEY -ErrorAction SilentlyContinue
    if ($TokenPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($TokenPointer)
    }
    $PlainToken = $null
    $SecureToken = $null
}

& $PythonExecutable -m server_training_pipeline.fetch_cmip6_member_resolved `
    status --root $CodeRoot | Tee-Object -FilePath (Join-Path $LogDirectory "status.log")
if ($LASTEXITCODE -ne 0) {
    throw "CMIP6 status reporting failed"
}
Write-Output "[$(Get-Date -Format o)] DONE Earth Data Hub CMIP6 gap recovery"
Write-Output "Logs: $LogDirectory"
