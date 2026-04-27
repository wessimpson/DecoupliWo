<#
.SYNOPSIS
  Launch 7 separate PowerShell processes, each running random-agent data collection (NumEnvs=1) for one game.

.DESCRIPTION
  Each process runs run_random_data_collection.ps1 with the default random agent:
  tracks.singlePlayer.simple.sampleRandom.Agent

  Default output (no -OutputRoot): <repo>/data/transitions/train/<game>/ — one folder per game stem.

.PARAMETER TotalTimesteps
  Frames per game (default: 100000).

.PARAMETER Scale
  Render scale for saved frames (default: 1.0).

.PARAMETER ChunkSize
  Rows per shard (default: 1000).

.PARAMETER Seed
  Optional RNG seed passed to each process.

.PARAMETER OutputRoot
  Optional; if set, passed to each run (games still get separate subfolders under that root).

.EXAMPLE
  .\run_random_parallel_7games.ps1

.EXAMPLE
  .\run_random_parallel_7games.ps1 -TotalTimesteps 200000 -Scale 0.5
#>
[CmdletBinding()]
param(
    [long] $TotalTimesteps = 100000,
    [double] $Scale = 0.5,
    [int] $ChunkSize = 1000,
    [Nullable[int]] $Seed = $null,
    [string] $OutputRoot = $null
)

$ErrorActionPreference = "Stop"

$games = @(
    "chopper",
    "defender",
    "jaws",
    "waves_rules_fast",
    "waves_rules_multishot",
    "waves_rules_ricochet",
    "waves"
)

$GvgaiRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runner = Join-Path $PSScriptRoot "run_random_data_collection.ps1"
if (-not (Test-Path $runner)) {
    Write-Error "Missing runner script: $runner"
}

Write-Host "Gvgai root: $GvgaiRoot"
Write-Host "Launching $($games.Count) random-agent collectors (NumEnvs=1 each), $TotalTimesteps timesteps per game."
Write-Host ""

foreach ($g in $games) {
    $argList = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $runner,
        $g,
        "-TotalTimesteps", "$TotalTimesteps",
        "-NumEnvs", "1",
        "-Scale", "$Scale",
        "-ChunkSize", "$ChunkSize"
    )
    if ($null -ne $Seed) { $argList += @("-Seed", "$Seed") }
    if ($OutputRoot) { $argList += @("-OutputRoot", $OutputRoot) }

    Write-Host "Starting: $g"
    Start-Process -FilePath "powershell.exe" -WorkingDirectory $GvgaiRoot -ArgumentList $argList
}

Write-Host ""
Write-Host "Launched $($games.Count) windows. Each writes to data\transitions\train\<game>\ (unless -OutputRoot set)."
