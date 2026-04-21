"""Interactive inference with user-key actions and generated frames only."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch

from world_model.ascii.constants import CANVAS_H, CANVAS_W, PAD_BYTE
from world_model.ascii.renderer import GvgaiRenderer
from world_model.ascii.tokenizer import dump_ascii, pad_to_canvas
from world_model.dataset import IMG_TRANSFORMS, crop_hw_div8
from world_model.model.net import ASCIIVAE
from world_model.model.world_model import Modality, WorldModel

DEFAULT_GVGAI_ROOT = Path("gvgai")


def _load_sd(path: Path, device: torch.device) -> dict:
	return torch.load(path, map_location=device, weights_only=True)


def _read_trainer_args(ckpt_dir: Path) -> dict[str, Any]:
	p = ckpt_dir / "trainer_state.pt"
	if not p.is_file():
		return {}
	blob = torch.load(p, map_location="cpu", weights_only=False)
	return dict(blob.get("args") or {})


def _coalesce(meta: dict[str, Any], key: str, override: Any, fallback: Any) -> Any:
	if override is not None:
		return override
	if key in meta and meta[key] is not None:
		return meta[key]
	return fallback


def load_world_model(
	ckpt_dir: Path,
	num_actions: int,
	history_len: int,
	vae_checkpoint: str | Path | None = None,
	pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
	modality: Modality | None = None,
) -> WorldModel:
	"""Load weights from ``ckpt_dir`` (same layout as :meth:`WorldModel.save_diffuser`).

	If ``trainer_state.pt`` exists, ``vae_checkpoint`` and ``modality`` default from saved args
	when omitted.
	"""
	ckpt_dir = Path(ckpt_dir)
	meta = _read_trainer_args(ckpt_dir)

	vae_eff = _coalesce(meta, "vae_checkpoint", vae_checkpoint, None)
	modality_eff: Modality = _coalesce(meta, "modality", modality, "pixel")
	wm = WorldModel(
		num_actions=num_actions,
		cross_attention_dim=768,
		vae_checkpoint=vae_eff,
		prediction_type="v_prediction",
		history_len=history_len,
		pretrained_model_name_or_path=pretrained_model_name_or_path,
		modality=modality_eff,
	)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	wm = wm.to(device)

	unet_path = ckpt_dir / "unet.pt"
	if not unet_path.is_file():
		raise FileNotFoundError(f"Missing UNet weights: {unet_path}")
	try:
		wm.diffuser.unet.load_state_dict(_load_sd(unet_path, device), strict=True)
	except RuntimeError as e:
		raise RuntimeError(
			f"UNet load failed from {unet_path}. "
			f"Ensure unet.pt matches this architecture (full SD1.4 UNet2D + widened conv_in)."
		) from e
	emb_path = ckpt_dir / "action_embedding.pt"
	if not emb_path.exists():
		emb_path = ckpt_dir / "future_action_embedding.pt"
	if emb_path.exists():
		wm.diffuser.action_embedding.load_state_dict(_load_sd(emb_path, device))
	for legacy_name in ("action_mlp.pt", "future_action_mlp.pt"):
		legacy = ckpt_dir / legacy_name
		if legacy.is_file():
			raise RuntimeError(
				f"Checkpoint has legacy {legacy_name} (MLP action head). Retrain with the current embedding-only diffuser."
			)
	return wm


def _tensor_to_imshow01(t: torch.Tensor) -> np.ndarray:
	"""[3,H,W] in [-1,1] → [H,W,3] float32 [0,1]."""
	return ((t.clamp(-1, 1) + 1) * 0.5).cpu().permute(1, 2, 0).numpy().astype(np.float32)


def run_autoregressive(
	ckpt_dir: str,
	env: str = "aliens",
	num_actions: int = 7,
	history_len: int = 2,
	num_inference_steps: int = 30,
	vae_checkpoint: Optional[str] = None,
) -> None:
	import matplotlib.gridspec as gridspec
	import matplotlib.pyplot as plt

	K = history_len
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	wm = load_world_model(
		Path(ckpt_dir), num_actions, K,
		vae_checkpoint=vae_checkpoint,
	)
	wm.eval()

	def tx(frame: np.ndarray) -> torch.Tensor:
		rgb = np.asarray(frame)[..., -3:]
		h, w = crop_hw_div8(*rgb.shape[:2])
		return IMG_TRANSFORMS(rgb[:h, :w])

	transitions_root = Path("data") / "transitions" / "train" / str(env)
	shards = sorted(p for p in transitions_root.glob("shard_*") if (p / "obs.npy").exists())
	assert shards, f"No shards under {transitions_root}"
	obs = np.load(shards[0] / "obs.npy", mmap_mode="r")
	acts = np.load(shards[0] / "action.npy", mmap_mode="r")
	n = int(obs.shape[0])
	assert n >= K + 1, f"Need at least K+1={K+1} frames, got {n}"

	# Bootstrap: first K real frames (same preprocessing as training).
	init = torch.stack([tx(np.asarray(obs[i])[..., -3:]) for i in range(K)], dim=0)
	gen_hist: deque[torch.Tensor] = deque([init[i].clone() for i in range(K)], maxlen=K)
	action_hist: deque[int] = deque([int(a) for a in acts[:K]], maxlen=K)
	data_pos = 0

	fig = plt.figure(figsize=(6, 4.5))
	gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.12], hspace=0.2, figure=fig)
	ax_top = fig.add_subplot(gs[0])
	text_ax = fig.add_subplot(gs[1])
	for ax in (ax_top, text_ax):
		ax.axis("off")
	preview = tx(np.asarray(obs[0])[..., -3:])
	h, w = int(preview.shape[-2]), int(preview.shape[-1])
	img_top = ax_top.imshow(np.zeros((h, w, 3), dtype=np.float32), vmin=0, vmax=1, interpolation="nearest")
	ax_top.set_title("generated (model)", fontsize=10)
	status = text_ax.text(
		0.5, 0.5, "Controls: arrows=move, space=shoot", ha="center", va="center", fontsize=10,
	)

	key_to_action = {
		"up": 1,
		"left": 2,
		"down": 3,
		"right": 4,
		" ": 5,
		"space": 5,
	}

	def on_key(event):
		nonlocal data_pos
		a_step = key_to_action.get(event.key)
		if a_step is None:
			return
		if data_pos + K >= n:
			status.set_text("End of trajectory (no more frames).")
			fig.canvas.draw_idle()
			return
		if a_step >= num_actions:
			status.set_text(f"Action {a_step} out of range for num_actions={num_actions}")
			fig.canvas.draw_idle()
			return

		ha = torch.tensor(list(action_hist), dtype=torch.long, device=device).view(1, K)
		fa = torch.tensor([a_step], dtype=torch.long, device=device)

		ctx = torch.stack(list(gen_hist), dim=0).unsqueeze(0).to(device)
		with torch.no_grad():
			z_hist = wm.encode_video(ctx)
			z_next = wm.generate_next_frame(
				z_hist, ha, fa, num_inference_steps=num_inference_steps,
			)
			gen = wm.decode_video(z_next)[0, 0].cpu()

		img_top.set_data(_tensor_to_imshow01(gen))

		gen_hist.append(gen.clone())
		action_hist.append(a_step)
		data_pos += 1
		status.set_text(
			f"step={data_pos}  window_start={data_pos - 1}  action={a_step}  "
			f"frames_left={n - data_pos - K}",
		)
		fig.canvas.draw_idle()

	fig.canvas.mpl_connect("key_press_event", on_key)
	plt.tight_layout()
	plt.show()


def _pad_ascii_frame(frame: np.ndarray) -> np.ndarray:
	return pad_to_canvas(np.ascontiguousarray(frame, dtype=np.uint8), CANVAS_H, CANVAS_W, PAD_BYTE)


def run_autoregressive_ascii(
	ckpt_dir: str,
	env: str = "aliens",
	num_actions: int = 7,
	history_len: int = 2,
	num_inference_steps: int = 30,
	vae_checkpoint: Optional[str] = None,
	render_pixels: bool = False,
	gvgai_root: Optional[Path] = None,
) -> None:
	"""Interactive ASCII-frame inference: show text prediction and (optionally) a GVGAI-rendered pixel frame."""
	import matplotlib.gridspec as gridspec
	import matplotlib.pyplot as plt

	K = history_len
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	wm = load_world_model(
		Path(ckpt_dir), num_actions, K,
		vae_checkpoint=vae_checkpoint, modality="ascii",
	)
	wm.train(False)

	transitions_root = Path("data") / "transitions" / "train" / str(env)
	shards = sorted(p for p in transitions_root.glob("shard_*") if (p / "obs.npy").exists())
	assert shards, f"No shards under {transitions_root}"
	obs = np.load(shards[0] / "obs.npy", mmap_mode="r")
	acts = np.load(shards[0] / "action.npy", mmap_mode="r")
	assert obs.dtype == np.uint8 and obs.ndim == 3, (
		f"ASCII inference expects uint8[N,H,W] obs, got dtype={obs.dtype} ndim={obs.ndim}"
	)
	n = int(obs.shape[0])
	assert n >= K + 1, f"Need at least K+1={K+1} frames, got {n}"

	init_frames = [torch.from_numpy(_pad_ascii_frame(np.asarray(obs[i]))).long() for i in range(K)]
	gen_hist: deque[torch.Tensor] = deque(init_frames, maxlen=K)
	action_hist: deque[int] = deque([int(a) for a in acts[:K]], maxlen=K)
	data_pos = 0

	renderer: Optional[GvgaiRenderer] = None
	if render_pixels:
		renderer = GvgaiRenderer(
			gvgai_root=Path(gvgai_root or DEFAULT_GVGAI_ROOT),
			game=env,
		)
		renderer.start()

	fig = plt.figure(figsize=(12, 5) if render_pixels else (8, 5))
	if render_pixels:
		gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.12], width_ratios=[1, 1], hspace=0.2, figure=fig)
		ax_text = fig.add_subplot(gs[0, 0])
		ax_pixels = fig.add_subplot(gs[0, 1])
		text_ax = fig.add_subplot(gs[1, :])
	else:
		gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.12], hspace=0.2, figure=fig)
		ax_text = fig.add_subplot(gs[0])
		ax_pixels = None
		text_ax = fig.add_subplot(gs[1])
	for ax in (ax_text, text_ax) + ((ax_pixels,) if ax_pixels is not None else ()):
		ax.axis("off")
	initial_dump = dump_ascii(init_frames[-1].numpy().astype(np.uint8))
	frame_text = ax_text.text(
		0.01, 0.99, initial_dump,
		ha="left", va="top", family="monospace", fontsize=10, transform=ax_text.transAxes,
	)
	ax_text.set_title("generated ASCII frame", fontsize=10)
	pixel_img = None
	if ax_pixels is not None and renderer is not None:
		first_rgb = renderer.render(init_frames[-1].numpy().astype(np.uint8))
		pixel_img = ax_pixels.imshow(first_rgb, interpolation="nearest")
		ax_pixels.set_title("rendered (GVGAI)", fontsize=10)
	status = text_ax.text(
		0.5, 0.5, "Controls: arrows=move, space=shoot", ha="center", va="center", fontsize=10,
	)

	key_to_action = {"up": 1, "left": 2, "down": 3, "right": 4, " ": 5, "space": 5}

	def on_key(event):
		nonlocal data_pos
		a_step = key_to_action.get(event.key)
		if a_step is None:
			return
		if data_pos + K >= n:
			status.set_text("End of trajectory (no more frames).")
			fig.canvas.draw_idle()
			return
		if a_step >= num_actions:
			status.set_text(f"Action {a_step} out of range for num_actions={num_actions}")
			fig.canvas.draw_idle()
			return

		ha = torch.tensor(list(action_hist), dtype=torch.long, device=device).view(1, K)
		fa = torch.tensor([a_step], dtype=torch.long, device=device)
		ctx = torch.stack(list(gen_hist), dim=0).unsqueeze(0).to(device)

		with torch.no_grad():
			z_hist = wm.encode_video(ctx)
			z_next = wm.generate_next_frame(
				z_hist, ha, fa, num_inference_steps=num_inference_steps,
			)
			logits = wm.decode_video(z_next)[0, 0]  # [V, H, W]
			pred_ids = ASCIIVAE.logits_to_ids(logits).to(torch.long).cpu()

		pred_np = pred_ids.numpy().astype(np.uint8)
		frame_text.set_text(dump_ascii(pred_np))
		if pixel_img is not None and renderer is not None:
			try:
				pixel_img.set_data(renderer.render(pred_np))
			except Exception as e:
				status.set_text(f"render error: {e}")
		gen_hist.append(pred_ids)
		action_hist.append(a_step)
		data_pos += 1
		status.set_text(
			f"step={data_pos}  window_start={data_pos - 1}  action={a_step}  "
			f"frames_left={n - data_pos - K}",
		)
		fig.canvas.draw_idle()

	fig.canvas.mpl_connect("key_press_event", on_key)

	def on_close(_event):
		if renderer is not None:
			renderer.close()

	fig.canvas.mpl_connect("close_event", on_close)
	plt.tight_layout()
	try:
		plt.show()
	finally:
		if renderer is not None:
			renderer.close()


def main() -> None:
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument("--ckpt_dir", type=str, default=str(Path("world_model") / "checkpoints" / "dit_encoded" / "step_0204960"))
	p.add_argument(
		"--vae_checkpoint",
		type=str,
		default=None,
		help="Path to vae.pt (empty = default based on --modality).",
	)
	p.add_argument("--env", type=str, default="aliens")
	p.add_argument("--num_inference_steps", type=int, default=10)
	p.add_argument("--num_actions", type=int, default=7)
	p.add_argument("--context_len", type=int, default=2, help="History length K (same as training).")
	p.add_argument("--modality", type=str, choices=["pixel", "ascii"], default="pixel")
	p.add_argument("--render_pixels", action="store_true",
		help="ASCII only: show GVGAI-rendered pixel preview next to the text output.")
	p.add_argument("--gvgai_root", type=str, default=str(DEFAULT_GVGAI_ROOT),
		help="Path to the gvgai submodule root (needed for --render_pixels).")
	args = p.parse_args()
	ckpt = args.ckpt_dir
	vae = args.vae_checkpoint.strip() if args.vae_checkpoint else None
	if args.modality == "ascii":
		run_autoregressive_ascii(
			ckpt_dir=ckpt,
			env=args.env,
			vae_checkpoint=vae,
			num_actions=args.num_actions,
			num_inference_steps=args.num_inference_steps,
			history_len=args.context_len,
			render_pixels=args.render_pixels,
			gvgai_root=Path(args.gvgai_root),
		)
	else:
		run_autoregressive(
			ckpt_dir=ckpt,
			env=args.env,
			vae_checkpoint=vae,
			num_actions=args.num_actions,
			num_inference_steps=args.num_inference_steps,
			history_len=args.context_len,
		)


if __name__ == "__main__":
	main()
