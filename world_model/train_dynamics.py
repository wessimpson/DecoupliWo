"""Deprecated entrypoint for the removed adversarial state-encoder trainer."""

from __future__ import annotations


def main() -> None:
	raise SystemExit(
		"world_model/train_dynamics.py was replaced by the residual correction architecture.\n"
		"Run pretraining with:\n"
		"  python -m world_model.train_original_dynamics ...\n"
		"Then train corrections with:\n"
		"  python -m world_model.train_residual_dynamics --base_ckpt_dir <base_step_dir> ..."
	)


if __name__ == "__main__":
	main()
