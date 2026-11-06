# Start (and optionally build) the local Wikipedia + E5 search service used by
# the search_qa environment (plan §1.2 / §4.3).
#
# Usage (Windows PowerShell, from anywhere):
#   # one-time: build the E5 index from a docs JSONL (Wikipedia dump converted)
#   .\agentic_rl_data\scripts\serve_search.ps1 -BuildIndex -Docs data\wiki\e5_docs.jsonl
#
#   # then serve (defaults: backend=e5, port=8899, prefix=<Docs prefix>)
#   .\agentic_rl_data\scripts\serve_search.ps1 -Docs data\wiki\e5_docs.jsonl
#
#   # smoke tests against a running service
#   python -m agentic_rl_data.envs.search health --url http://127.0.0.1:8899
#   python -m agentic_rl_data.envs.search bench --url http://127.0.0.1:8899
#
# build-index writes <prefix>.npz (+<prefix>.faiss if faiss is installed),
# <prefix>.docs.jsonl and <prefix>.meta.json; serve accepts either index file.
# Dependencies: `pip install fastapi uvicorn numpy` (+ faiss-cpu + sentence
# transformers / torch for the e5 backend).

param(
    [ValidateSet("inmemory", "e5")][string]$Backend = "e5",
    [string]$Docs = "",
    [string]$Index = "",
    [string]$Model = "intfloat/e5-base-v2",
    [int]$Port = 8899,
    [string]$HostBind = "0.0.0.0",
    [int]$Workers = 1,
    [double]$RequestTimeout = 5.0,
    [switch]$BuildIndex,
    [string]$OutPrefix = "",
    [string]$Device = ""
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Push-Location $root
try {
    $python = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }

    if ($BuildIndex) {
        if (-not $Docs) { throw "-BuildIndex requires -Docs <docs.jsonl>" }
        $prefix = if ($OutPrefix) { $OutPrefix } else { "$Docs.index" }
        $indexArgs = @(
            "-m", "agentic_rl_data.envs.search", "build-index",
            "--docs", $Docs,
            "--out-prefix", $prefix,
            "--model", $Model
        )
        if ($Device) { $indexArgs += @("--device", $Device) }
        & $python @indexArgs
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        if (-not $Index) {
            $Index = if (Test-Path "$prefix.faiss") { "$prefix.faiss" } else { "$prefix.npz" }
        }
        if (-not $Docs) { $Docs = "$prefix.docs.jsonl" }
        Write-Host "[serve_search] built index: $Index"
    }

    $serveArgs = @(
        "-m", "agentic_rl_data.envs.search", "serve",
        "--backend", $Backend,
        "--port", "$Port",
        "--host", $HostBind,
        "--workers", "$Workers",
        "--model", $Model,
        "--request-timeout", "$RequestTimeout"
    )
    if ($Docs) { $serveArgs += @("--docs", $Docs) }
    if ($Index) { $serveArgs += @("--index", $Index) }
    if ($Device) { $serveArgs += @("--device", $Device) }

    & $python @serveArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
