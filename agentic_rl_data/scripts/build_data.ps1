# Build the Agentic RL dataset end to end (plan §3 steps 1-8).
#
# Usage (Windows PowerShell, from anywhere):
#   .\agentic_rl_data\scripts\build_data.ps1                 # run with default config
#   .\agentic_rl_data\scripts\build_data.ps1 -DryRun         # print the plan, write nothing
#   .\agentic_rl_data\scripts\build_data.ps1 -Strict         # fail-closed report gate (plan §8)
#   .\agentic_rl_data\scripts\build_data.ps1 -Config path\to\pipeline.json
#
# Prerequisites: Python 3.10+, `datasets` for downloading sources (or point
# build_raw.cfg_json at local files), and optionally `bfcl-eval` data for the
# decontamination step (otherwise that step is reported as skipped/[NOT RUN]).

param(
    [string]$Config = "",
    [switch]$DryRun,
    [switch]$Strict,
    [switch]$KeepGoing
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Push-Location $root
try {
    if (-not $Config) {
        $Config = "agentic_rl_data/configs/pipeline.json"
    }
    $python = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }

    $pipelineArgs = @(
        "-m", "agentic_rl_data.scripts.run_pipeline",
        "--config", $Config
    )
    if ($DryRun) { $pipelineArgs += "--dry-run" }
    if ($Strict) { $pipelineArgs += "--strict-report" }
    if ($KeepGoing) { $pipelineArgs += "--keep-going" }

    & $python @pipelineArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
