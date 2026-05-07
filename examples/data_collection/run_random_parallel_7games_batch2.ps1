<#
.SYNOPSIS
  Launch 7 separate PowerShell processes for random-agent collection — complement to run_random_parallel_7games.ps1.

.DESCRIPTION
  Games: aliens, aliens_rules_*, chopper_rules_* (see $games below).
  Same defaults as run_random_parallel_7games.ps1 (100000 timesteps, NumEnvs=1, etc.).

.EXAMPLE
  .\run_random_parallel_7games_batch2.ps1
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
    "aliens",
    "aliens_rules_multishot",
    "aliens_rules_ricochet",
    "chopper_rules_multishot",
    "chopper_rules_ricochet"
)

$GvgaiRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runner = Join-Path $PSScriptRoot "run_random_data_collection.ps1"
if (-not (Test-Path $runner)) {
    Write-Error "Missing runner script: $runner"
}

Write-Host "Gvgai root: $GvgaiRoot"
Write-Host "Launching $($games.Count) random-agent collectors (batch 2), $TotalTimesteps timesteps per game."
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
    if ($g -eq "zelda") {
        $argList += @("-Agent", "tracks.singlePlayer.advanced.olets.Agent")
    }
    if ($null -ne $Seed) { $argList += @("-Seed", "$Seed") }
    if ($OutputRoot) { $argList += @("-OutputRoot", $OutputRoot) }

    Write-Host "Starting: $g"
    Start-Process -FilePath "powershell.exe" -WorkingDirectory $GvgaiRoot -ArgumentList $argList
}

Write-Host ""
Write-Host "Launched $($games.Count) windows. Each writes to data\transitions\train\<game>\ (unless -OutputRoot set)."
