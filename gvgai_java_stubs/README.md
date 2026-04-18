# GVGAI Java stubs — ASCII render server

This folder holds Java source that needs to live inside the `gvgai` submodule
but is drafted here so it is committable from the main repo. Once the
colleague owning the submodule accepts the change, the file should be moved
into the submodule's matching path and the class re-compiled with GVGAI.

## File

- `src/tracks/singlePlayer/rendering/AsciiRenderServer.java`
  - Package: `tracks.singlePlayer.rendering`
  - Depends on: `core.game.Game`, `core.vgdl.VGDLParser`, `core.vgdl.VGDLFactory`,
    `core.vgdl.VGDLRegistry`, `core.vgdl.VGDLSprite`.
    All exist in the `gvgai-aliens-train-variants` branch.

## Install into the submodule

```sh
cd <repo-root>
git submodule update --init --recursive gvgai
mkdir -p gvgai/src/tracks/singlePlayer/rendering
cp gvgai_java_stubs/src/tracks/singlePlayer/rendering/AsciiRenderServer.java \
   gvgai/src/tracks/singlePlayer/rendering/AsciiRenderServer.java
```

## Build

The existing GVGAI project compiles with IntelliJ's `out/production/gvgai`
layout. Either recompile the whole project from IntelliJ/ant, or compile just
this class against the existing output:

```sh
cd gvgai
javac \
  -cp out/production/gvgai \
  -d out/production/gvgai \
  src/tracks/singlePlayer/rendering/AsciiRenderServer.java
```

## Run standalone (sanity test)

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
