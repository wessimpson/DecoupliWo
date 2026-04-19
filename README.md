# Custom Pong

## Install

```bash
python -m pip install pygame-ce
```

## Run

```bash
python main.py --mode normal
```

Modes: `normal` | `gravity` | `teleport`

## Controls

`Up` move up  
`Down` move down  
`R` reset  
`1 2 3` switch mode  
`Esc` quit

## Test

```bash
python -m unittest discover -s tests -v
```

## Headless

Use `render_mode=None` in `PongEnv(...)`.

## Transition Data + State-Based Editable World Model

Use the `gvgai_jpype` env or any Python env with `numpy` and `torch`.

Run the smallest end-to-end sanity check with only one environment and one rule, `pong:normal`:

```bash
./scripts/pong_normal_smoke.sh
```

This collects only Pong normal transitions, trains a small model, evaluates only normal mode with `--eval-modes normal`, and runs headless model playback.

Collect broad counterfactual transitions. This stores the same state/action under all three rule variants:

```bash
/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python data/collect_pong_transitions.py \
  --output data/transitions/custom_pong/counterfactual_mixed_seed0 \
  --episodes 5000 \
  --steps-per-episode 200 \
  --policy mixed \
  --counterfactual \
  --val-fraction 0.1 \
  --seed 0
```

Append targeted rare and diverse transitions to the same dataset:

```bash
/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python data/collect_pong_transitions.py \
  --output data/transitions/custom_pong/counterfactual_mixed_seed0 \
  --episodes 0 \
  --rare-events \
  --rare-samples-per-source 20000 \
  --val-fraction 0.1 \
  --seed 1
```

Rare sources include `diverse`, `left_wall`, `top_bounce`, `bottom_bounce`, `wrapped_top`, `wrapped_bottom`, `paddle_hit`, and `miss`.

Each shard stores both the original flat Pong state and a fixed object-slot representation:

```text
state, action, next_state, rule_id, game_id
object_slots, object_mask, next_object_slots, next_object_mask
```

For Pong, slot `0` is the ball, slot `1` is the paddle, and remaining slots are inactive. This is the state-based editable world-model interface; no visual encoder or pixel decoder is used yet.

Collect a mixed Pong + Breakout-lite dataset in the same shared-slot format:

```bash
/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python data/collect_editable_world_transitions.py \
  --output data/transitions/editable_world/pong_breakout_counterfactual_seed0 \
  --games pong breakout \
  --episodes 5000 \
  --steps-per-episode 300 \
  --policy mixed \
  --counterfactual \
  --val-fraction 0.1 \
  --seed 0
```

Breakout-lite uses the same object schema: slot `0` is the ball, slot `1` is the paddle, and slots `2-9` are blocks. The model learns `next_object_mask` so disappearing blocks are represented.

Generate a larger diverse dataset with random, heuristic, and mixed policies plus targeted rare states:

```bash
./scripts/generate_large_editable_world_dataset.sh
```

Defaults:

```text
OUTPUT=data/transitions/editable_world/pong_breakout_large_seed0
GAMES="pong breakout"
MODES="normal gravity teleport"
POLICIES="random heuristic mixed"
EPISODES=5000
STEPS_PER_EPISODE=300
RARE_SAMPLES_PER_SOURCE=20000
```

The large generator combines ordinary rollouts with targeted rare starts. For Pong this includes wall bounces, top/bottom bounces, teleport wraps, paddle hits, misses, and diverse random states. For Breakout this includes left/right wall cases, top bounces, teleport wraps, paddle hits, block hits, misses, and diverse random states.

Train PPO+RND and save rollout transitions:

```bash
/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python data/train_pong_ppo_rnd.py \
  --output data/transitions/custom_pong/ppo_rnd_seed0 \
  --logdir runs/pong_ppo_rnd_seed0 \
  --total-steps 500000 \
  --rnd-scale 0.2 \
  --seed 0
```

Inspect a dataset:

```bash
/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python data/inspect_pong_transitions.py \
  data/transitions/custom_pong/counterfactual_mixed_seed0 \
  --split train
```

Train the rule-conditioned GNN dynamics model:

```bash
/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python train_pong_world_model.py \
  --dataset data/transitions/editable_world/pong_breakout_counterfactual_seed0 \
  --output runs/editable_world_slot_gnn_seed0 \
  --epochs 100 \
  --batch-size 1024 \
  --device auto
```

Train while holding out a rule combination from optimization and reporting held-out validation metrics:

```bash
/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python train_pong_world_model.py \
  --dataset data/transitions/editable_world/pong_breakout_counterfactual_seed0 \
  --output runs/editable_world_slot_gnn_holdout_seed0 \
  --epochs 100 \
  --batch-size 1024 \
  --holdout-combos pong:teleport breakout:gravity \
  --device auto
```

Run the rule-use ablation by training separate models with `--rule-ablation zero` or `--rule-ablation shuffle`, then compare their eval JSON against the normal run:

```bash
/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python train_pong_world_model.py \
  --dataset data/transitions/custom_pong/counterfactual_mixed_seed0 \
  --output runs/pong_rule_gnn_ablation_zero_seed0 \
  --epochs 100 \
  --batch-size 1024 \
  --rule-ablation zero \
  --device auto
```

Evaluate one-step, rollout, and counterfactual rule-conditioning metrics:

```bash
/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python eval_pong_world_model.py \
  --checkpoint runs/editable_world_slot_gnn_seed0/best.pt \
  --dataset data/transitions/editable_world/pong_breakout_counterfactual_seed0 \
  --output runs/editable_world_slot_gnn_seed0/eval.json \
  --holdout-combos pong:teleport breakout:gravity \
  --device auto
```

Play through the learned world model using pygame:

```bash
/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python play_pong_world_model.py \
  --checkpoint runs/editable_world_slot_gnn_seed0/best.pt \
  --mode gravity \
  --device auto \
  --random-start \
  --seed 123
```

Controls: `Up`/`Down` move the paddle, `R` resets, `1`/`2`/`3` switch rules, `P` pauses, `S` single-steps, and `Esc` quits.

Play Breakout-lite through the same learned world model:

```bash
/home/soyuj/miniconda3/envs/gvgai_jpype/bin/python play_breakout_world_model.py \
  --checkpoint runs/editable_world_slot_gnn_seed0/best.pt \
  --mode teleport \
  --device auto \
  --random-start \
  --seed 123
```

Breakout controls: `Left`/`Right` move the paddle, `R` resets, `1`/`2`/`3` switch rules, `P` pauses, and `Esc` quits.
