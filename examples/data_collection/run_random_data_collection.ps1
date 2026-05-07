<#
.SYNOPSIS
  Run data collection with GVGAI sample random agent on a game from examples/data_collection.

.DESCRIPTION
  Resolves the gvgai project root (this script lives in gvgai/examples/data_collection),
  compiles sources if needed, then runs tracks.singlePlayer.RunDataCollectionAgent.

  Transitions are written under <repo>/data/transitions/train/<Game>/.

  Default agent: tracks.singlePlayer.simple.sampleRandom.Agent

.PARAMETER Game
  Short file stem (e.g. chopper_rules_multishot) or path to a .txt game under gvgai.

.PARAMETER Level
  Optional level file path (relative to gvgai root unless absolute).

.PARAMETER Agent
  Optional fully qualified agent class name (defaults to random agent).

.PARAMETER TotalTimesteps
  Total frames to collect across episodes. If omitted, runs one episode.

.PARAMETER NumEnvs
  Number of parallel environments (default: 1).

.PARAMETER Visuals
  Switch: open the Swing window (default: headless).

.PARAMETER OutputRoot
  Optional directory for transition shards.

.PARAMETER ChunkSize
  Optional rows per shard (default: 1000 when omitted; same as Java collector).

.PARAMETER Seed
  Optional integer RNG seed.

.PARAMETER Scale
  Render scale for saved observations (default: 1.0 = actual full-resolution RGB frame).

.PARAMETER List
  Print available game stems (same as: java ... RunDataCollectionAgent --list).

.EXAMPLE
  .\run_random_data_collection.ps1 aliens

.EXAMPLE
  .\run_random_data_collection.ps1 waves_rules_multishot -TotalTimesteps 200000 -NumEnvs 8
#>
[CmdletBinding(DefaultParameterSetName = "Run")]
param(
    [Parameter(Mandatory = $true, Position = 0, ParameterSetName = "Run")]
    [string] $Game,

    [Parameter(Mandatory = $false)]
    [string] $Level,

    [Parameter(Mandatory = $false)]
    [string] $Agent = "tracks.singlePlayer.simple.sampleRandom.Agent",

    [Parameter(Mandatory = $false)]
    [Nullable[long]] $TotalTimesteps,

    [Parameter(Mandatory = $false)]
    [int] $NumEnvs = 1,

    [switch] $Visuals,

    [Parameter(Mandatory = $false)]
    [string] $OutputRoot,

    [Parameter(Mandatory = $false)]
    [Nullable[int]] $ChunkSize,

    [Parameter(Mandatory = $false)]
    [Nullable[int]] $Seed,

    [Parameter(Mandatory = $false)]
    [double] $Scale = 1.0,

    [Parameter(Mandatory = $true, ParameterSetName = "List")]
    [switch] $List
)

$ErrorActionPreference = "Stop"
$GvgaiRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $GvgaiRoot

$outDir = Join-Path $GvgaiRoot "out"
$gson = Join-Path $GvgaiRoot "gson-2.6.2.jar"
$sourcesList = Join-Path $GvgaiRoot "sources_build.txt"

if (-not (Test-Path $gson)) {
    Write-Error "Missing gson jar: $gson"
}

if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

$agentClass = Join-Path $outDir "tracks\singlePlayer\RunDataCollectionAgent.class"
$needsCompile = -not (Test-Path $agentClass)
if (-not $needsCompile) {
    $latestJava = Get-ChildItem -Path (Join-Path $GvgaiRoot "src") -Recurse -Filter "*.java" |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -ne $latestJava) {
        $needsCompile = $latestJava.LastWriteTimeUtc -gt (Get-Item $agentClass).LastWriteTimeUtc
    }
}

if ($needsCompile) {
    Write-Host "Compiling gvgai (first run or sources changed)..."
    Get-ChildItem -Path (Join-Path $GvgaiRoot "src") -Recurse -Filter "*.java" | ForEach-Object { $_.FullName } | Set-Content -Path $sourcesList -Encoding ASCII
    & javac --release 8 -encoding UTF-8 -d $outDir -cp $gson "@$sourcesList"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$cp = "$outDir;$gson"
if ($List) {
    & java -cp $cp tracks.singlePlayer.RunDataCollectionAgent --list
    exit $LASTEXITCODE
}

$strongAgent = "tracks.singlePlayer.advanced.olets.Agent"
$gameStem = [System.IO.Path]::GetFileNameWithoutExtension($Game)
$resolvedAgent = $Agent
if (($gameStem -eq "zelda") -and (-not $PSBoundParameters.ContainsKey("Agent"))) {
    $resolvedAgent = $strongAgent
}

$javaArgs = @("tracks.singlePlayer.RunDataCollectionAgent", "--game", $Game, "--agent", $resolvedAgent, "--num-envs", "$NumEnvs")
if ($Level) { $javaArgs += @("--level", $Level) }
if ($null -ne $TotalTimesteps) { $javaArgs += @("--total-timesteps", "$TotalTimesteps") }
if ($Visuals) { $javaArgs += "--visuals" }
if ($OutputRoot) { $javaArgs += @("--output-root", $OutputRoot) }
if ($null -ne $ChunkSize) { $javaArgs += @("--chunk-size", "$ChunkSize") }
if ($null -ne $Seed) { $javaArgs += @("--seed", "$Seed") }
$javaArgs += @("--scale", "$Scale")

Write-Host "cwd: $GvgaiRoot"
Write-Host "java -cp `"$cp`" $($javaArgs -join ' ')"
& java -cp $cp @javaArgs
exit $LASTEXITCODE
