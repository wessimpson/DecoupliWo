#!/bin/bash
#SBATCH --account=torch_pr_611_general
#SBATCH --job-name=wm-dynamics
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

set -euo pipefail

module purge 2>/dev/null || true
unset LD_LIBRARY_PATH

# Before submitting, run once from the repo root:
#   bash scripts/setup_hpc_venv.sh
#   mkdir -p logs
#
# Examples:
#   STAGE=original VENV_DIR=.venv sbatch scripts/world_model_dynamics.sbatch
#   STAGE=residual BASE_CKPT_DIR=world_model/checkpoints/original_dynamics/step_0050000 VENV_DIR=.venv sbatch scripts/world_model_dynamics.sbatch
#   STAGE=analyze BASE_CKPT_DIR=... RESIDUAL_CKPT_DIR=... VENV_DIR=.venv sbatch scripts/world_model_dynamics.sbatch
#
# Common overrides:
#   ENV_NAME=aliens              # optional; omit for all eligible folders
#   TRANSITIONS_ROOT=/path/data/transitions
#   ENCODED_SUBDIR=encoded       # training reads $TRANSITIONS_ROOT/$ENCODED_SUBDIR/train
#   VAE_CHECKPOINT=/path/world_model/checkpoints/vae/vae.pt
#   BATCH_SIZE=8 MAX_TRAIN_STEPS=50000 NUM_WORKERS=4
#   VALIDATION_EVERY=0           # use this if you only copied encoded/train and no encoded/test
#   EXTRA_ARGS="--validation_every 5000 --save_every 5000"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"

if [[ ! -d world_model || ! -f requirements.txt ]]; then
	echo "REPO_ROOT does not look like the DecoupliWo repo: $REPO_ROOT" >&2
	echo "Submit from /scratch/sjb8193/DecoupliWo or set REPO_ROOT=/scratch/sjb8193/DecoupliWo." >&2
	exit 2
fi

mkdir -p \
	logs \
	runs/world_model_original \
	runs/world_model_residual \
	world_model/checkpoints/original_dynamics \
	world_model/checkpoints/residual_dynamics \
	analysis/residual_transfer

if [[ -n "${MODULES:-}" ]]; then
	# Example: MODULES="cuda/12.1 gcc/11"
	module load ${MODULES}
fi

if [[ -n "${VENV_DIR:-}" ]]; then
	source "$VENV_DIR/bin/activate"
elif [[ -d ".venv" ]]; then
	source ".venv/bin/activate"
elif [[ -n "${CONDA_ENV:-}" ]]; then
	source "$(conda info --base)/etc/profile.d/conda.sh"
	conda activate "$CONDA_ENV"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NVML_BASED_CUDA_CHECK=1

PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${STAGE:-original}"
TRANSITIONS_ROOT="${TRANSITIONS_ROOT:-data/transitions}"
ENCODED_SUBDIR="${ENCODED_SUBDIR:-encoded}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-world_model/checkpoints/vae/vae.pt}"
NUM_ACTIONS="${NUM_ACTIONS:-7}"
CONTEXT_LEN="${CONTEXT_LEN:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50000000}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
LR="${LR:-5e-5}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
VALIDATION_EVERY="${VALIDATION_EVERY:-10000}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
VAL_SAMPLES="${VAL_SAMPLES:-8}"
VAL_AR_HORIZONS="${VAL_AR_HORIZONS:-1,10,30}"
TRANSFER_PROBE_ENVS="${TRANSFER_PROBE_ENVS:-defender,jaws}"
TRANSFER_PROBE_RULE="${TRANSFER_PROBE_RULE:-multishot}"
TRANSFER_PROBE_ACTION="${TRANSFER_PROBE_ACTION:-5}"
TRANSFER_PROBE_STEPS="${TRANSFER_PROBE_STEPS:-10}"
ORIGINAL_CHECKPOINT_DIR="${ORIGINAL_CHECKPOINT_DIR:-world_model/checkpoints/original_dynamics}"
RESIDUAL_CHECKPOINT_DIR="${RESIDUAL_CHECKPOINT_DIR:-world_model/checkpoints/residual_dynamics}"
ORIGINAL_LOG_DIR="${ORIGINAL_LOG_DIR:-runs/world_model_original}"
RESIDUAL_LOG_DIR="${RESIDUAL_LOG_DIR:-runs/world_model_residual}"
NORMAL_ANCHOR_RATIO="${NORMAL_ANCHOR_RATIO:-0.25}"
ANALYSIS_OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-analysis/residual_transfer}"
MAX_ANALYSIS_SAMPLES="${MAX_ANALYSIS_SAMPLES:-512}"

echo "repo=$REPO_ROOT"
echo "stage=$STAGE"
echo "python=$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
if [[ "${CUDA_PREFLIGHT:-0}" == "1" ]]; then
	"$PYTHON_BIN" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
	print("cuda_device", torch.cuda.get_device_name(0))
PY
else
	"$PYTHON_BIN" - <<'PY'
import torch
print("torch", torch.__version__)
PY
fi

common_args=(
	--transitions_root "$TRANSITIONS_ROOT"
	--encoded_subdir "$ENCODED_SUBDIR"
	--vae_checkpoint "$VAE_CHECKPOINT"
	--num_actions "$NUM_ACTIONS"
	--context_len "$CONTEXT_LEN"
	--batch_size "$BATCH_SIZE"
	--max_train_steps "$MAX_TRAIN_STEPS"
	--num_train_epochs "$NUM_TRAIN_EPOCHS"
	--lr "$LR"
	--num_workers "$NUM_WORKERS"
	--mixed_precision "$MIXED_PRECISION"
	--validation_every "$VALIDATION_EVERY"
	--save_every "$SAVE_EVERY"
	--num_inference_steps "$NUM_INFERENCE_STEPS"
	--val_samples "$VAL_SAMPLES"
	--val_ar_horizons "$VAL_AR_HORIZONS"
	--transfer_probe_envs "$TRANSFER_PROBE_ENVS"
	--transfer_probe_rule "$TRANSFER_PROBE_RULE"
	--transfer_probe_action "$TRANSFER_PROBE_ACTION"
	--transfer_probe_steps "$TRANSFER_PROBE_STEPS"
)

env_args=()
if [[ -n "${ENV_NAME:-}" ]]; then
	env_args=(--env "$ENV_NAME")
fi

gradient_args=()
if [[ "${GRADIENT_CHECKPOINTING:-0}" == "1" ]]; then
	gradient_args=(--gradient_checkpointing)
fi

extra_args=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
	# shellcheck disable=SC2206
	extra_args=(${EXTRA_ARGS})
fi

case "$STAGE" in
	original)
		cmd=(
			"$PYTHON_BIN" -m world_model.train_original_dynamics
			"${common_args[@]}"
			"${env_args[@]}"
			"${gradient_args[@]}"
			--checkpoint_dir "$ORIGINAL_CHECKPOINT_DIR"
			--log_dir "$ORIGINAL_LOG_DIR"
			"${extra_args[@]}"
		)
		;;
	residual)
		: "${BASE_CKPT_DIR:?Set BASE_CKPT_DIR to an original_dynamics/step_XXXXXXX checkpoint directory}"
		cmd=(
			"$PYTHON_BIN" -m world_model.train_residual_dynamics
			--base_ckpt_dir "$BASE_CKPT_DIR"
			"${common_args[@]}"
			"${env_args[@]}"
			"${gradient_args[@]}"
			--normal_anchor_ratio "$NORMAL_ANCHOR_RATIO"
			--checkpoint_dir "$RESIDUAL_CHECKPOINT_DIR"
			--log_dir "$RESIDUAL_LOG_DIR"
			"${extra_args[@]}"
		)
		;;
	analyze)
		: "${BASE_CKPT_DIR:?Set BASE_CKPT_DIR to an original_dynamics/step_XXXXXXX checkpoint directory}"
		: "${RESIDUAL_CKPT_DIR:?Set RESIDUAL_CKPT_DIR to a residual_dynamics/step_XXXXXXX checkpoint directory}"
		cmd=(
			"$PYTHON_BIN" -m world_model.analyze_residual_transfer
			--base_ckpt_dir "$BASE_CKPT_DIR"
			--residual_ckpt_dir "$RESIDUAL_CKPT_DIR"
			--transitions_root "$TRANSITIONS_ROOT"
			--encoded_subdir "$ENCODED_SUBDIR"
			--vae_checkpoint "$VAE_CHECKPOINT"
			--num_actions "$NUM_ACTIONS"
			--context_len "$CONTEXT_LEN"
			--batch_size "$BATCH_SIZE"
			--num_workers "$NUM_WORKERS"
			--max_samples "$MAX_ANALYSIS_SAMPLES"
			--output_dir "$ANALYSIS_OUTPUT_DIR"
			"${env_args[@]}"
			"${extra_args[@]}"
		)
		;;
	*)
		echo "Unknown STAGE=$STAGE; expected original, residual, or analyze" >&2
		exit 2
		;;
esac

printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
