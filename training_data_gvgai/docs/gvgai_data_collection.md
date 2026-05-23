# GVGAI Data Collection

This repository now has a variant-aware rollout format for GVGAI world-model training.

## Source Engine

Use the vendored JPype fork in `training_data_gvgai/gvgai/`:

```bash
python training_data_gvgai/gvgai/build.py
pip install -r training_data_gvgai/requirements.txt
pip install -e training_data_gvgai/gvgai
```

The collector can use either registered `gym_gvgai` env IDs or explicit VGDL files. Explicit VGDL files are preferred for variants because every variant can point at a different game-rule file while reusing the same level files. Fork provenance is recorded in `training_data_gvgai/gvgai/FORK_PROVENANCE.md`.

## Dataset Layout

Collection writes:

```text
training_data_gvgai/data/gvgai_rollouts/
  train/
    aliens/
      default/
        manifest.jsonl
        shard_00000/
          obs.npy
          next_obs.npy
          action.npy
          reward.npy
          terminated.npy
          truncated.npy
          episode_id.npy
          step_in_episode.npy
          level.npy
          seed.npy
          avatar_xy.npy
          rule_flags.npy
          active_rule_flags.npy
          frame_metadata.jsonl
          n_actions.npy
          metadata.json
```

Each row is one transition: `obs_t`, `action_t`, `reward_t`, `next_obs_t`, terminal flags, provenance, and numeric rule vectors. `rule_flags.npy` stores the configured rollout-level flags for backwards compatibility. `active_rule_flags.npy` stores frame-level labels for the same columns, where labels only turn on while the rule is visually or causally expressed. The order of both rule-flag arrays is stored in each shard’s `metadata.json`.

`frame_metadata.jsonl` has one JSON object per transition. It is the embedding-friendly record for the frame/transition and includes:

- `game`, `level`, `rollout_variant`, `configured_variants`, and `active_variants`
- `action.raw` and normalized `action.semantic` values such as `move_left`, `move_right`, `move_up`, `move_down`, `fire_projectile`, and `idle`
- `frame.ascii.grid`, a one-character-per-cell grid, plus `frame.ascii.legend`, `frame.ascii.raw`, score, winner state, block size, and typed object positions extracted from the simulator state
- tick-level collision/event records from GVGAI, with labels such as `enemy.hit_by_projectile` and `player.object_collision`
- `rule_flags_static` and `rule_flags_active`

This is intentionally transition-aligned: the metadata row describes `obs_t` plus the action that produces `next_obs_t`. For variants like `multishot`, the frame-level flag activates on `fire_projectile` and remains active while player-created projectile objects are visible.

Compact ASCII uses `.` for empty cells and a stable single-character legend for visible sprite classes, for example `@` for the avatar, `E` for enemies, `#` for bases/walls, `|` for player projectiles, and `*` for bombs. The original comma-separated GVGAI ASCII is preserved in `frame.ascii.raw`.

## Frame Metadata JSON Schema

Saved `frame_metadata.jsonl` rows use this shape:

```json
{
  "schema_version": "gvgai-frame-metadata-v1",
  "game": "aliens",
  "level": 0,
  "rollout_variant": "physics_a",
  "configured_variants": ["physics_a"],
  "active_variants": ["physics_a"],
  "episode_id": 0,
  "step_in_episode": 47,
  "seed": 1813382118,
  "tick": 46,
  "next_tick": 47,
  "action": {
    "id": 3,
    "raw": "ACTION_RIGHT",
    "semantic": "move_right"
  },
  "labels": {
    "player": {
      "making": "move_right",
      "receiving": []
    },
    "enemy": {
      "making": [],
      "receiving": []
    },
    "global_disturbance": []
  },
  "frame": {
    "ascii": {
      "encoding": "single_char_grid_v1",
      "grid": "..EE..E...EE...\n###..........EE",
      "raw": "original GVGAI comma-separated grid",
      "legend": {
        ".": ["empty"],
        "@": ["avatar"],
        "E": ["alienBlue"],
        "#": ["base"]
      },
      "empty": "."
    },
    "score": 1.0,
    "winner": "NO_WINNER",
    "block_size": 32,
    "objects": {}
  },
  "events": [],
  "rule_flags_static": {
    "physics_a": 1.0
  },
  "rule_flags_active": {
    "physics_a": 1.0
  }
}
```

Field notes:

- `configured_variants` means the rollout was generated from a rule file with those variants enabled.
- `active_variants` means the variant is active for this transition. Always-on physics/control variants are active every frame; projectile and hit-animation variants activate only while their visual or event window is present.
- `labels.player.making` is derived from the chosen GVGAI action. `labels.player.receiving`, `labels.enemy.receiving`, and `events` are only populated when the engine exposes an event that can be classified reliably.
- `frame.objects` is optional diagnostic state. Some GVGAI paths currently return `{}` for the detailed state JSON, so consumers should not require this field. The compact ASCII grid is the reliable spatial representation today.

## Compact ASCII Representation

`frame.ascii.grid` is the model-facing spatial text representation. It has exactly one character per board cell and newline-separated rows. It is easier to read and embed than GVGAI's raw comma-separated cell strings.

Default character policy:

| Character | Meaning |
| --- | --- |
| `.` | empty cell |
| `@` | avatar/player |
| `E` | enemy-like sprite, such as `alienBlue` |
| `#` | base, wall, shield, or other barrier-like sprite |
| `|` | player projectile, missile, bullet, or shot |
| `*` | bomb or explosive projectile |
| `O` | portal |
| `$` | resource |
| `a-z`, `0-9` | fallback characters for game-specific sprite types |

Each row also carries `frame.ascii.legend`, so consumers should read the legend instead of assuming every fallback character has a global meaning. When multiple sprites occupy one cell, the grid picks a single display character by priority: avatar, projectiles, bombs, enemies, barriers, portals, resources, then fallback sprites. The original GVGAI observation string remains available at `frame.ascii.raw` for auditing.

Example:

```text
..EE..E...EE...
###..........EE
###..........EE
...............
...###...###...
......@........
```

## Variant Catalog

`training_data_gvgai/data/gvgai_variant_catalog.example.json` defines 3 games × 4 variants:

- Games: `aliens`, `chopper`, `waves`
- Variants: `default`, `physics_a`, `physics_b`, `physics_c`

Add more variants by creating a new catalog entry with:

- `game_file`: VGDL game-rule file for that variant
- `level_files`: level files used with that rule file
- `rule_flags`: numeric conditioning flags the world model can consume at inference
- `variant_label_rules`: optional frame-label activation rules, keyed by rule flag name

Example:

```json
{
  "game_file": "aliens_multishot.txt",
  "level_files": ["aliens_lvl0.txt"],
  "rule_flags": {"multishot": 1, "enemy_explode": 1},
  "variant_label_rules": {
    "multishot": {"activation": "player_projectile_window"},
    "enemy_explode": {"activation": "enemy_hit_window", "grace_frames": 4}
  }
}
```

Supported activation rules today:

- `always`: active for every frame in the rollout, useful for global physics or control disturbances.
- `player_projectile_window`: active when the player fires or while player-created projectile sprites are present.
- `enemy_hit_window`: active on simulator events classified as `enemy.hit_by_projectile`, with optional `grace_frames` for explosion/color animations.

If a rule flag has no explicit `variant_label_rules`, the collector applies conservative name-based defaults for common labels such as `multishot`, `split_orthogonal`, `ricochet`, `shoot_walls`, `enemy_explode`, `two_hit_color`, and global names containing `physics`, `control`, or `disturbance`.

## Collect

Small smoke collection:

```bash
python -m training_data_gvgai.data.collect_gvgai_jpype \
  --games aliens \
  --variants default \
  --split train \
  --frames-per-variant 100 \
  --chunk-size 50 \
  --policy repeat_random
```

Live metadata verification in the terminal:

```bash
python -m training_data_gvgai.data.collect_gvgai_jpype \
  --games aliens \
  --variants multishot \
  --frames-per-variant 200 \
  --metadata-preview summary \
  --metadata-preview-every 1
```

Use `--metadata-preview json` to print the full embedding record for each sampled frame, or `--metadata-preview ascii` to show the compact summary plus the ASCII grid. Increase `--metadata-preview-every` for longer runs, for example `--metadata-preview-every 10`.

## Live Web Preview

Use the web preview when you want to visually audit the running game, compact ASCII grid, and cleaned metadata side by side:

```bash
python -m training_data_gvgai.data.preview_gvgai_live \
  --game aliens \
  --variant physics_a \
  --level 0 \
  --policy repeat_random \
  --step-delay 0.18 \
  --port 8769
```

Then open:

```text
http://127.0.0.1:8769
```

The web interface shows:

- left: the rendered GVGAI frame
- middle/right: a compact ASCII panel with its legend
- right: a cleaned JSON preview

The cleaned browser JSON intentionally omits `frame.ascii` because the ASCII grid is already shown in a separate panel. It also hides empty diagnostic fields that are not populated reliably by the current bridge. A typical browser JSON record looks like:

```json
{
  "game": "aliens",
  "level": 0,
  "variant": "physics_a",
  "episode": {
    "id": 0,
    "seed": 1813382118,
    "step": 47
  },
  "tick": {
    "current": 46,
    "next": 47
  },
  "action": {
    "id": 3,
    "raw": "ACTION_RIGHT",
    "semantic": "move_right"
  },
  "variants": {
    "configured": ["physics_a"],
    "active": ["physics_a"],
    "active_flags": {
      "physics_a": 1.0
    }
  },
  "labels": {
    "player_action": "move_right"
  },
  "score": 1.0,
  "winner": "NO_WINNER"
}
```

Use a different port if one is already occupied. The current catalog supports `aliens`, `chopper`, and `waves`, each with `default`, `physics_a`, `physics_b`, and `physics_c`. Example:

```bash
python -m training_data_gvgai.data.preview_gvgai_live --game chopper --variant physics_a --port 8770
```

Full 3 × 4 train collection:

```bash
python -m training_data_gvgai.data.collect_gvgai_jpype \
  --split train \
  --frames-per-variant 1000000 \
  --chunk-size 5000 \
  --policy repeat_random
```

Use separate seeds and splits for validation/test:

```bash
python -m training_data_gvgai.data.collect_gvgai_jpype --split validation --seed 1000 --frames-per-variant 50000
python -m training_data_gvgai.data.collect_gvgai_jpype --split test --seed 2000 --frames-per-variant 50000
```

## Export For Hugging Face

Build a dataset-repo folder:

```bash
python -m training_data_gvgai.data.export_gvgai_hf \
  --data-root training_data_gvgai/data/gvgai_rollouts \
  --export-dir training_data_gvgai/data/hf_gvgai_dataset \
  --dataset-name "GVGAI variant rollouts"
```

Upload when authenticated:

```bash
HF_TOKEN=... python -m training_data_gvgai.data.export_gvgai_hf \
  --data-root training_data_gvgai/data/gvgai_rollouts \
  --export-dir training_data_gvgai/data/hf_gvgai_dataset \
  --repo-id your-org/gvgai-variant-rollouts \
  --upload \
  --private
```

The export keeps the large NumPy files in Git LFS and publishes `manifest.jsonl` as the stable index. This is more practical than one giant Parquet table for high-volume video data; Parquet can be added later for low-resolution previews or metadata-only queries.

## Conditioning Model Contract

The collection layer gives the world model two conditioning axes:

- `game`: categorical ID such as `aliens`, `chopper`, `waves`
- `rule_flags`: configured rollout-level dense vector such as `[physics_a, physics_b, physics_c]`
- `active_rule_flags`: frame-level dense vector with the same column order, used when the model should learn only the visible/causal interval for a variant

Inference should accept the same flag names and order used during training. Changing flags at inference is then a model-conditioning problem, not a data-loading problem. The JSONL metadata can also be embedded directly when training a text/metadata-conditioned model.
