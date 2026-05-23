# GVGAI Data Collection

This repository has a variant-aware rollout format for GVGAI world-model training.

## Source Engine

Use the vendored JPype fork in `training_data_gvgai/gvgai/`:

```bash
python training_data_gvgai/gvgai/build.py
pip install -r training_data_gvgai/requirements.txt
pip install -e training_data_gvgai/gvgai
```

The collector can use either registered `gym_gvgai` env IDs or explicit VGDL files. The checked-in world-model catalog uses registered env IDs that point at `games_world_model/` stems.

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
          n_actions.npy
          metadata.json
```

Each row is one transition: `obs_t`, `action_t`, `reward_t`, `next_obs_t`, terminal flags, provenance, and a numeric `rule_flags` vector. The order of the rule-flag columns is stored in each shard's `metadata.json`.

## Variant Catalog

The default collector catalog is `training_data_gvgai/data/gvgai_variant_catalog.world_model.json`.
It mirrors the checked-in `games_world_model/` tree and currently covers 13 base GVGAI games
with their default rules plus 101 rule variants.

Each variant entry stores:

- `env_id`: registered `gym_gvgai` env stem for that rule file
- `levels`: level IDs shared by the base game directory
- `rule_flags`: numeric conditioning flags the world model can consume at inference

`training_data_gvgai/data/gvgai_variant_catalog.example.json` remains as a small legacy example with the older `aliens` / `chopper` / `waves` physics variants.

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

Full catalog train collection:

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

- `game`: categorical base-game ID such as `aliens`, `defender`, `pacman`
- `rule_flags`: dense vector keyed by rule tags such as `multishot`, `shield_reflect`, or `quick_dash_3tile`

Inference should accept the same flag names and order used during training. Changing flags at inference is then a model-conditioning problem, not a data-loading problem.
