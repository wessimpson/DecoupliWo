# GVGAI Java stubs

This folder holds Java source that needs to live inside the `gvgai` submodule
but is drafted here so it is committable from the main repo. Once the
colleague owning the submodule accepts the change, each file should be moved
into the submodule's matching path and the class re-compiled with GVGAI.

Two independent subsystems live here:

1. **ASCII render server** (`src/tracks/singlePlayer/rendering/AsciiRenderServer.java`)
   — a TCP service consumed by `world_model/ascii/renderer.py` that turns an
   ASCII grid into a PNG for inference-time visualization.
2. **ASCII data collector** (`src/tracks/singlePlayer/ascii/*.java`) — a
   headless MCTS driver that writes `data/transitions/{train,test}/<game>/shard_*`
   shards for `world_model/train_ascii_vae.py`. Python wrapper:
   `data/collect_gvgai_ascii.py`.

## Files

- `src/tracks/singlePlayer/rendering/AsciiRenderServer.java`
  - Package: `tracks.singlePlayer.rendering`
  - Depends on: `core.game.Game`, `core.vgdl.VGDLParser`, `core.vgdl.VGDLFactory`,
    `core.vgdl.VGDLRegistry`, `core.vgdl.VGDLSprite`.
- `src/tracks/singlePlayer/ascii/AsciiGridExtractor.java`
- `src/tracks/singlePlayer/ascii/NpyWriter.java`
- `src/tracks/singlePlayer/ascii/RecordingMCTSAgent.java`
- `src/tracks/singlePlayer/ascii/RunAsciiCollectionMCTS.java`
  - Package: `tracks.singlePlayer.ascii`
  - Depends on: `core.game.StateObservation`, `core.game.Observation`,
    `core.vgdl.VGDLRegistry`, `ontology.Types`, `tools.ElapsedCpuTimer`,
    `tools.Vector2d`, `tracks.ArcadeMachine`,
    `tracks.singlePlayer.advanced.sampleMCTS.{Agent, SingleMCTSPlayer}`.

## Install into the submodule

```sh
cd <repo-root>
# If you haven't yet, clone GVGAI into the (empty) gvgai/ folder:
#   git clone https://github.com/GAIGResearch/GVGAI.git gvgai
mkdir -p gvgai/src/tracks/singlePlayer/rendering gvgai/src/tracks/singlePlayer/ascii
cp gvgai_java_stubs/src/tracks/singlePlayer/rendering/AsciiRenderServer.java \
   gvgai/src/tracks/singlePlayer/rendering/AsciiRenderServer.java
cp gvgai_java_stubs/src/tracks/singlePlayer/ascii/*.java \
   gvgai/src/tracks/singlePlayer/ascii/
```

## Build

The existing GVGAI project compiles with IntelliJ's `out/production/gvgai`
layout. Either recompile the whole project from IntelliJ/ant, or compile just
the stub classes against the existing output:

```sh
cd gvgai
javac \
  -cp out/production/gvgai \
  -d out/production/gvgai \
  src/tracks/singlePlayer/rendering/AsciiRenderServer.java \
  src/tracks/singlePlayer/ascii/*.java
```

If `out/production/gvgai` does not exist yet (fresh clone, no IntelliJ build),
bootstrap it with:

```sh
cd gvgai
mkdir -p out/production/gvgai
javac -d out/production/gvgai $(find src -name "*.java")
```

## Collect ASCII training data

The `RunAsciiCollectionMCTS` main class runs the stock GVGAI MCTS agent
headless, tokenizes each tick's `StateObservation` to ASCII using the JSON
mapping in `world_model/ascii/mappings/<game>.json`, and writes shards in the
exact layout `world_model/train_ascii_vae.py::AllAsciiFramesDataset` expects.
The preferred way to invoke it is the Python wrapper, which also handles the
train/test split:

```sh
cd <repo-root>
python -m data.collect_gvgai_ascii \
  --games aliens,chopper,waves \
  --frames-per-game 500000 \
  --mcts-ms 40
```

Or call the Java class directly for one game/split:

```sh
cd <repo-root>/gvgai
java -cp out/production/gvgai tracks.singlePlayer.ascii.RunAsciiCollectionMCTS \
  --gvgai-root .. /gvgai \
  --repo-root  .. \
  --game       aliens \
  --out        ../data/transitions/train/aliens \
  --frames     50000 \
  --mcts-ms    40 \
  --levels     0,1,2,3,4 \
  --chunk-size 5000 \
  --seed       42
```

Output per shard (all files are NumPy v1.0 `.npy`):

- `obs.npy` — `uint8[N, H, W]` ASCII bytes (one char per grid cell).
- `action.npy` — `int64[N]` action ordinals from `ontology.Types.ACTIONS`
  (`NIL, UP, LEFT, DOWN, RIGHT, USE, ESCAPE`, matching
  `data/view_transition_shard.py::GVGAI_ACTION_LABELS`).
- `n_actions.npy` — `int64` scalar, always `7`.
- `reward.npy` — `float32[N]` per-tick score delta.
- `done.npy` — `uint8[N]`, `1` only on the last frame of each episode.
- `player_x.npy`, `player_y.npy` — `float32[N]` avatar position in pixels
  (`NaN` if there is no avatar in that frame).

## Run the render server standalone (sanity test)

```sh
cd gvgai
java -cp out/production/gvgai tracks.singlePlayer.rendering.AsciiRenderServer 0
# prints e.g. "AsciiRenderServer listening on 54321"
```

Then in another shell:

```sh
nc 127.0.0.1 54321 <<'EOF'
INIT examples/gridphysics/aliens.txt
RENDER 0 11
1.............................
000...........................
000...........................
..............................
..............................
..............................
..............................
....000......000000.....000...
...00000....00000000...00000..
...0...0....00....00...00000..
................A.............
QUIT
EOF
```

You should see an `OK <bytes>` header followed by raw PNG data in stdout.

## How the Python side talks to it

`world_model/ascii/renderer.py` (`GvgaiRenderer`) spawns the server with
`subprocess.Popen`, parses the "listening on" line from stdout to find the
bound port, connects a TCP socket, and speaks the same protocol:

```
INIT <abs_path_to_vgdl>
OK <screen_w> <screen_h> <block_size>

RENDER 0 <num_rows>
<row_0>
...
<row_{N-1}>
OK <png_bytes_len>
<raw png bytes>

QUIT
BYE
```

`GVGAI_CLASSPATH` env var overrides the default classpath
(`<gvgai_root>/out/production/gvgai`) if you put the build output elsewhere.

## Notes / limitations

- Rendering reflects sprite position but not the orientation of sprites that
  move (the ASCII char doesn't encode heading). Good enough for visualization.
- One `Game` object is cached per `INIT` path per connection; `RENDER`
  re-uses that parsed game. Re-issue `INIT` with a different VGDL path to
  switch games on the same connection.
- The server is single-threaded: one concurrent client at a time. Adequate
  for inference preview and batched validation rendering; if you need
  parallelism later, run multiple server processes on different ports.
