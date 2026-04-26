#!/usr/bin/env bash
#SBATCH --account=torch_pr_611_general
#SBATCH --job-name=gns-shared-full-medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/sjb8193/game_wm/logs/slurm/%x-%j.out
#SBATCH --error=/scratch/sjb8193/game_wm/logs/slurm/%x-%j.err

set -euo pipefail

ROOT="${ROOT:-/scratch/sjb8193/game_wm/DecoupliWo}"
cd "$ROOT"

mkdir -p /scratch/sjb8193/game_wm/logs/slurm

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/scratch/sjb8193/penv}"
export PYTHON="${PYTHON:-$CONDA_ENV_PREFIX/bin/python}"
export SEED="${SEED:-0}"
export DATASET="${DATASET:-$ROOT/data/transitions/gns_shared/pong_breakout_seed${SEED}}"
export OUTPUT="${OUTPUT:-$ROOT/runs/gns_shared_full_medium_seed${SEED}}"
export DEVICE="${DEVICE:-cuda:0}"
export EPOCHS="${EPOCHS:-200}"
export BATCH_SIZE="${BATCH_SIZE:-7168}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

echo "ROOT=$ROOT"
echo "DATASET=$DATASET"
echo "OUTPUT=$OUTPUT"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "DEVICE=$DEVICE"

"$PYTHON" train_gns_shared_simulator.py \
  --dataset "$DATASET" \
  --output "$OUTPUT" \
  --model-size medium \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --num-workers "$NUM_WORKERS" \
  --train-combos pong:normal pong:gravity breakout:normal breakout:teleport \
  --holdout-combos pong:teleport breakout:gravity
