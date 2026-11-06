# Quick smoke check for the agentic_rl_data stack (Windows PowerShell).
#
#   .\agentic_rl_data\scripts\smoke.ps1              # tests + pipeline dry-run
#   .\agentic_rl_data\scripts\smoke.ps1 -SkipTests   # only the pipeline dry-run
#
# It never starts training and never writes data (dry-run only).  A running
# search service can be checked separately:
#   python -m agentic_rl_data.envs.search health --url http://127.0.0.1:8899

param(
    [string]$Config = "",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Push-Location $root
try {
    $python = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
    if (-not $Config) { $Config = "agentic_rl_data/configs/pipeline.json" }

    if (-not $SkipTests) {
        Write-Host "[smoke] pytest agentic_rl_data/tests -q"
        & $python -m pytest agentic_rl_data/tests -q
        if ($LASTEXITCODE -ne 0) {
            Write-Error "[smoke] pytest failed"
            exit $LASTEXITCODE
        }
    }

    Write-Host "[smoke] pipeline dry-run (no files written)"
    & $python -m agentic_rl_data.scripts.run_pipeline --config $Config --dry-run
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "[smoke] OK"
}
finally {
    Pop-Location
}
