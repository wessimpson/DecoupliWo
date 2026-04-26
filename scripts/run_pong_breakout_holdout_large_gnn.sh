#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# One-command full pipeline:
# 1. Generate the large Pong+Breakout dataset.
# 2. Train the large rule-conditioned GNN with compositional holdouts.
#
# Defaults:
#   dataset -> /T7/users/soyuj/runs/data/transitions/editable_world/pong_breakout_large_seed0
#   output  -> /T7/users/soyuj/runs/pong_breakout_holdout_large_gnn_seed0
#
# Override examples:
#   SEED=1 DEVICE=cuda:0 ./scripts/run_pong_breakout_holdout_large_gnn.sh
#   RUN_ROOT=/some/other/drive ./scripts/run_pong_breakout_holdout_large_gnn.sh

SKIP_DATASET=0 bash scripts/train_pong_breakout_holdout_large_gnn.sh
