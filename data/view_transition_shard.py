from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


DEFAULT_TRANSITIONS_ROOT = Path("data") / "transitions"
DEFAULT_ENV = "train"
DEFAULT_SPLIT = "aliens_rules_fast"

GVGAI_ACTION_LABELS = [
	"ACTION_NIL",
	"ACTION_UP",
	"ACTION_LEFT",
	"ACTION_DOWN",
	"ACTION_RIGHT",
	"ACTION_USE",
	"ACTION_ESCAPE",
]

ATARI_ACTION_LABELS = [
	"NOOP",
	"FIRE",
	"UP",
	"RIGHT",
	"LEFT",
	"DOWN",
	"UPRIGHT",
	"UPLEFT",
	"DOWNRIGHT",
	"DOWNLEFT",
	"UPFIRE",
	"RIGHTFIRE",
	"LEFTFIRE",
	"DOWNFIRE",
	"UPRIGHTFIRE",
	"UPLEFTFIRE",
	"DOWNRIGHTFIRE",
	"DOWNLEFTFIRE",
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="View transition frames one-by-one. Press space to advance, q/escape to quit. "
		"Works with both Atari (collect_transitions.py) and GVGAI (RunDataCollectionAgent) shards.",
	)
	parser.add_argument(
		"--trainthis-game",
		type=str,
		default=None,
		metavar="STEM",
		help="Use trainthis/<STEM>/shard_* (Java RunDataCollectionAgent). Overrides --env/--split.",
	)
	parser.add_argument(
		"--env",
		type=str,
		default=DEFAULT_ENV,
		help=f"Split subdirectory under {DEFAULT_TRANSITIONS_ROOT} (default: {DEFAULT_ENV}).",
	)
	parser.add_argument(
		"--split",
		type=str,
		default=DEFAULT_SPLIT,
		help=f"Dataset split to inspect (default: {DEFAULT_SPLIT}).",
	)
	parser.add_argument(
		"--shard",
		type=str,
		default=None,
		help="Specific shard directory name, e.g. shard_00700. Defaults to the first shard in the split.",
	)
	parser.add_argument(
		"--shard_dir",
		type=str,
		default=None,
		help="Full path to a shard directory (e.g. trainthis/aliens/shard_00000). Overrides other path options.",
	)
	parser.add_argument(
		"--start",
		type=int,
		default=0,
		help="Initial timestep to display.",
	)
	parser.add_argument(
		"--frame_skip",
		type=int,
		default=1,
		help="Frame skip to apply to the video.",
	)
	parser.add_argument(
		"--downsample",
		type=int,
		default=1,
		help="Downsample display frames by this factor using high-quality antialiasing. 1 keeps the original size.",
	)
	return parser.parse_args()


def resolve_shard_dir(args: argparse.Namespace) -> Path:
	if args.shard_dir is not None:
		shard_dir = Path(args.shard_dir)
		if not shard_dir.exists():
			raise FileNotFoundError(f"Shard directory not found: {shard_dir}")
		return shard_dir

	if args.trainthis_game is not None:
		split_dir = Path("trainthis") / args.trainthis_game.strip()
		if not split_dir.exists():
			raise FileNotFoundError(f"trainthis game directory not found: {split_dir}")
		shards = sorted(
			path
			for path in split_dir.glob("shard_*")
			if (path / "obs.npy").exists() and (path / "action.npy").exists()
		)
		if not shards:
			raise FileNotFoundError(f"No shard_* with obs/action under {split_dir}")
		if args.shard is not None:
			shard_dir = split_dir / args.shard
			if not shard_dir.exists():
				raise FileNotFoundError(f"Shard directory not found: {shard_dir}")
			return shard_dir
		return shards[0]

	split_dir = DEFAULT_TRANSITIONS_ROOT / args.env / args.split
	if not split_dir.exists():
		raise FileNotFoundError(f"Split directory not found: {split_dir}")

	if args.shard is not None:
		shard_dir = split_dir / args.shard
		if not shard_dir.exists():
			raise FileNotFoundError(f"Shard directory not found: {shard_dir}")
		return shard_dir

	shards = sorted(
		path
		for path in split_dir.glob("shard_*")
		if (path / "obs.npy").exists() and (path / "action.npy").exists()
	)
	if not shards:
		raise FileNotFoundError(f"No shard_* directories with obs/action arrays found under {split_dir}")
	return shards[0]


# ---------------------------------------------------------------------------
# Frame preparation
# ---------------------------------------------------------------------------

def ensure_displayable_frame(frame: np.ndarray) -> np.ndarray:
	frame = np.asarray(frame)
	if frame.ndim != 3:
		raise ValueError(f"Expected frame with shape [H, W, C], got {frame.shape}")
	if frame.shape[-1] > 3:
		frame = frame[..., -3:]
	if frame.dtype != np.uint8:
		if np.issubdtype(frame.dtype, np.floating) and frame.size and frame.max() <= 1.0:
			frame = (frame * 255.0).round()
		frame = np.clip(frame, 0, 255).astype(np.uint8)
	return frame


def maybe_downsample_frame(frame: np.ndarray, downsample: int) -> np.ndarray:
	downsample = max(1, int(downsample))
	if downsample == 1:
		return frame

	height, width = frame.shape[:2]
	target_width = max(1, width // downsample)
	target_height = max(1, height // downsample)

	img = Image.fromarray(frame)
	even = width == target_width * downsample and height == target_height * downsample
	resample = Image.Resampling.BOX if even else Image.Resampling.LANCZOS
	return np.asarray(img.resize((target_width, target_height), resample))


def format_action(action_value: int | None, n_actions: int | None = None) -> str:
	if action_value is None:
		return "N/A"
	if n_actions == len(GVGAI_ACTION_LABELS) and 0 <= action_value < len(GVGAI_ACTION_LABELS):
		return f"{action_value} ({GVGAI_ACTION_LABELS[action_value]})"
	if n_actions == len(ATARI_ACTION_LABELS) and 0 <= action_value < len(ATARI_ACTION_LABELS):
		return f"{action_value} ({ATARI_ACTION_LABELS[action_value]})"
	if 0 <= action_value < len(ATARI_ACTION_LABELS):
		return f"{action_value} ({ATARI_ACTION_LABELS[action_value]})"
	return str(action_value)


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

def view_shard(
	shard_dir: Path,
	start_idx: int = 0,
	frame_skip: int = 1,
	downsample: int = 1,
) -> None:
	try:
		import matplotlib.pyplot as plt
	except ImportError as exc:
		raise ImportError(
			"view_transition_shard requires matplotlib to display frames. "
			"Install it in your environment, then rerun the script."
		) from exc

	obs = np.load(shard_dir / "obs.npy", mmap_mode="r")
	actions = np.load(shard_dir / "action.npy", mmap_mode="r")
	n_actions_path = shard_dir / "n_actions.npy"
	n_actions_meta: int | None = None
	if n_actions_path.exists():
		n_actions_meta = int(np.load(n_actions_path))
	num_frames = int(obs.shape[0])
	if num_frames == 0:
		raise ValueError(f"No frames found in {shard_dir / 'obs.npy'}")

	_, obs_h, obs_w = obs.shape[0], obs.shape[1], obs.shape[2]

	def make_frame(idx: int) -> np.ndarray:
		raw = np.array(obs[idx])
		return maybe_downsample_frame(ensure_displayable_frame(raw), downsample)

	start_idx = max(0, min(int(start_idx), num_frames - 1))
	first_frame = make_frame(start_idx)
	disp_h, disp_w = first_frame.shape[:2]

	fig_w = max(6.0, disp_w / 80.0)
	fig_h = max(4.0, disp_h / 80.0 + 1.0)
	fig, (image_ax, text_ax) = plt.subplots(
		2, 1, figsize=(fig_w, fig_h),
		gridspec_kw={"height_ratios": [18, 1.5], "hspace": 0.08},
	)
	image_ax.axis("off")
	text_ax.axis("off")

	image_artist = image_ax.imshow(first_frame, interpolation="bilinear")
	status_text = text_ax.text(0.5, 0.5, "", ha="center", va="center", fontsize=10)
	index_state = {"value": start_idx}

	def update_display() -> None:
		idx = index_state["value"]
		frame = make_frame(idx)
		image_artist.set_data(frame)
		action_value = int(actions[idx]) if idx < len(actions) else None
		res_str = f"{obs_w}x{obs_h}"
		if downsample > 1:
			res_str += f" (display {disp_w}x{disp_h})"
		status_text.set_text(
			f"timestep: {idx} / {num_frames - 1}    "
			f"action: {format_action(action_value, n_actions_meta)}    "
			f"resolution: {res_str}"
		)
		fig.canvas.draw_idle()

	def on_key(event) -> None:
		if event.key in {" ", "space"}:
			if index_state["value"] < num_frames - frame_skip:
				index_state["value"] += frame_skip
				update_display()
		elif event.key in {"escape", "q"}:
			plt.close(fig)

	fig.canvas.mpl_connect("key_press_event", on_key)
	try:
		fig.canvas.manager.set_window_title(f"Transition Viewer - {shard_dir.name}")
	except Exception:
		pass

	print(f"Viewing shard: {shard_dir}")
	print(f"  obs shape: {obs.shape}  (WxH = {obs_w}x{obs_h})")
	print(f"  frames: {num_frames}  n_actions: {n_actions_meta}")
	print("Controls: space = next frame, q/escape = quit")
	update_display()
	plt.show()


def main() -> None:
	args = parse_args()
	shard_dir = resolve_shard_dir(args)
	view_shard(
		shard_dir,
		start_idx=args.start,
		frame_skip=args.frame_skip,
		downsample=args.downsample,
	)


if __name__ == "__main__":
	main()
