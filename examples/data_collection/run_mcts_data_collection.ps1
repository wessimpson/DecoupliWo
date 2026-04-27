<#
.SYNOPSIS
  Run the GVGAI sample MCTS agent on a game from examples/data_collection (headless by default).

.DESCRIPTION
  Resolves the gvgai project root (this script lives in gvgai/examples/data_collection),
  compiles sources if needed, then runs tracks.singlePlayer.RunDataCollectionAgent.

  Transitions are written under <repo>/data/transitions/train/<Game>/.

  Default agent: tracks.singlePlayer.advanced.sampleMCTS.Agent (UCT MCTS in advanced/sampleMCTS/Agent.java)

.PARAMETER Game
  Short file stem (e.g. chopper_rules_multishot) or path to a .txt game under gvgai.

.PARAMETER Level
  Optional level file path (relative to gvgai root unless absolute).

.PARAMETER Agent
  Optional fully qualified agent class name.

.PARAMETER Visuals
  Switch: open the Swing window (default: headless).

.PARAMETER OutputRoot
  Optional directory for transition shards (default: ../trainthis relative to gvgai cwd).

.PARAMETER ChunkSize
  Optional rows per shard (default: 1000 when omitted; same as Java collector).

.PARAMETER Seed
  Optional integer RNG seed.

.PARAMETER Scale
  Render scale for saved observations (default: 1.0 = actual full-resolution RGB frame).

.PARAMETER List
  Print available game stems (same as: java ... RunDataCollectionAgent --list).

.EXAMPLE
  .\run_mcts_data_collection.ps1 chopper_rules_multishot

.EXAMPLE
  .\run_mcts_data_collection.ps1 aliens_rules_multishot -Seed 0 -ChunkSize 1000

.EXAMPLE
  .\run_mcts_data_collection.ps1 waves -Agent tracks.singlePlayer.advanced.sampleRHEA.Agent -Visuals
#>
[CmdletBinding(DefaultParameterSetName = "Run")]
param(
    [Parameter(Mandatory = $true, Position = 0, ParameterSetName = "Run")]
    [string] $Game,

    [Parameter(Mandatory = $false)]
    [string] $Level,

    [Parameter(Mandatory = $false)]
    [string] $Agent,

    [switch] $Visuals,

    [Parameter(Mandatory = $false)]
    [string] $OutputRoot,

    [Parameter(Mandatory = $false)]
    [Nullable[int]] $ChunkSize,

    [Parameter(Mandatory = $false)]
    [Nullable[int]] $Seed,

    [Parameter(Mandatory = $false)]
    [double] $Scale = 0.5,

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

$javaArgs = @("tracks.singlePlayer.RunDataCollectionAgent", "--game", $Game)
if ($Level) { $javaArgs += @("--level", $Level) }
if ($Agent) { $javaArgs += @("--agent", $Agent) }
if ($Visuals) { $javaArgs += "--visuals" }
if ($OutputRoot) { $javaArgs += @("--output-root", $OutputRoot) }
if ($null -ne $ChunkSize) { $javaArgs += @("--chunk-size", "$ChunkSize") }
if ($null -ne $Seed) { $javaArgs += @("--seed", "$Seed") }
$javaArgs += @("--scale", "$Scale")

Write-Host "cwd: $GvgaiRoot"
Write-Host "java -cp `"$cp`" $($javaArgs -join ' ')"
& java -cp $cp @javaArgs
exit $LASTEXITCODE
