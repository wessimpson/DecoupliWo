"""
Train dynamics (diffusion UNet) on pre-VAE-encoded transition shards.

Same loop as ``train_world_model.py``, but batches load ``history_latents`` / ``target_latent``
from ``data/transitions/encoded/...`` (see ``encode_transition.py``). VAE is still loaded for
decode-only validation (PSNR / previews).

VRAM tips: lower ``--batch_size``; pass ``--gradient_checkpointing``; shorter ``--val_ar_horizons``; raise
``--validation_every`` to val less often. Autoregressive val runs the rollout for ``max(val_ar_horizons)``
steps (same DDIM step count as training ``--num_inference_steps``). Only predicted latents at horizon
boundaries (``--val_ar_horizons``) are kept; VAE decodes just those timesteps. ``val/psnr_ar/hXX`` is
PSNR on the **last frame at that horizon** (pred vs GT at rollout index ``h-1``), not the full prefix.
TensorBoard also logs ``val/lpips_ar/hXX`` for the same frame (LPIPS on ``[-1,1]``). AR rollouts cache **one
window per encoded test folder** (env+rule variant) when possible; images are under
``val/ar_rollout/hXX/<base_game>`` with target above generated per rule column, rule columns side-by-side
when a base game has multiple variants.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from world_model.dataset import (
	MixedEncodedRolloutVideoDataset,
	canonical_rule_onehots,
	encoded_dirs_all_under_split,
	encoded_dirs_with_rules,
	preprocess_latent,
)
from world_model.model.error_buffer import ErrorBuffer
from world_model.model.world_model import WorldModel

CONTEXT_LEN = 4
CROSS_ATTENTION_DIM = 768
PREDICTION_TYPE = "v_prediction"
PRETRAINED_MODEL_NAME_OR_PATH = "CompVis/stable-diffusion-v1-4"
DEFAULT_CHECKPOINT_DIR_RULES = Path("world_model") / "checkpoints" / "dit_encoded_rules"
DEFAULT_CHECKPOINT_DIR_ALL_ENV = Path("world_model") / "checkpoints" / "dit_encoded_rules_all_env"
DEFAULT_CHECKPOINT_DIR_RULES_ADV = Path("world_model") / "checkpoints" / "dit_encoded_rules_adv"
DEFAULT_CHECKPOINT_DIR_ALL_ENV_ADV = Path("world_model") / "checkpoints" / "dit_encoded_rules_all_env_adv"


def psnr_neg1_to_01(pred: torch.Tensor, tgt: torch.Tensor) -> float:
	p = ((pred.clamp(-1, 1) + 1) * 0.5).float()
	t = ((tgt.clamp(-1, 1) + 1) * 0.5).float()
	mse = (p - t).pow(2).mean().item()
	if mse <= 0:
		return float("inf")
	return 10.0 * math.log10(1.0 / mse)


def future_residuals_as_history_block(delta_bn: torch.Tensor, K: int) -> torch.Tensor:
	if delta_bn.dim() == 4:
		delta_bn = delta_bn.unsqueeze(1)
	B, N, C, h, w = delta_bn.shape
	if N == K:
		return delta_bn
	if N > K:
		return delta_bn[:, :K]
	pad = delta_bn[:, -1:].expand(B, K - N, C, h, w)
	return torch.cat([delta_bn, pad], dim=1)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Train dynamics on encoded transition latents.")
	p.add_argument(
		"--env",
		type=str,
		default=None,
		help="Base name under encoded/{train,test}/ (e.g. aliens). If omitted, use every subdirectory "
		"that contains encoded shards. Rule one-hot follows folder name suffix (_rules_fast, etc.); "
		"same suffix across games shares the same rule conditioning.",
	)
	p.add_argument("--transitions_root", type=str, default=str(Path("data") / "transitions"))
	p.add_argument("--encoded_subdir", type=str, default="encoded", help="Under transitions_root, same as encode script.")
	p.add_argument(
		"--vae_checkpoint",
		type=str,
		default="",
		help="Optional local Wan VAE state dict. Empty uses pretrained Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae.",
	)
	p.add_argument("--num_actions", type=int, default=7)
	p.add_argument("--context_len", type=int, default=CONTEXT_LEN)
	p.add_argument(
		"--batch_size",
		type=int,
		default=8,
		help="Training batch size; also chunks validation (diffusion MSE, generate_next_frame, VAE decode, AR decode).",
	)
	p.add_argument("--num_train_epochs", type=int, default=5)
	p.add_argument("--max_train_steps", type=int, default=50000_000)
	p.add_argument("--lr", type=float, default=5e-5)
	p.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
	p.add_argument("--lr_warmup_steps", type=int, default=500)
	p.add_argument("--log_dir", type=str, default="runs/world_model_dynamics_adv")
	p.add_argument("--num_inference_steps", type=int, default=10)
	p.add_argument("--gradient_checkpointing", action="store_true")
	p.add_argument("--gamma", type=float, default=0.1)
	p.add_argument("--gamma_warmup_steps", type=int, default=500)
	p.add_argument("--error_buffer_cap", type=int, default=5_000)
	p.add_argument("--validation_every", type=int, default=10_000)
	p.add_argument(
		"--checkpoint_dir",
		type=str,
		default=None,
		help="Checkpoint root. Default: *_adv folder for this training variant "
		"(all envs -> dit_encoded_rules_all_env_adv; single env -> dit_encoded_rules_adv).",
	)
	p.add_argument("--save_every", type=int, default=10_000)
	p.add_argument("--max_grad_norm", type=float, default=1.0)
	p.add_argument(
		"--val_samples",
		type=int,
		default=8,
		help="Validation windows sampled per rule (normal / fast / multishot / ricochet); missing rules are skipped.",
	)
	p.add_argument("--num_workers", type=int, default=4)
	p.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="bf16")
	p.add_argument(
		"--val_ar_horizons",
		type=str,
		default="1,10,30",
		help="Comma-separated rollout horizons; rollout length is max(...). Latents are stored and "
		"VAE-decoded only at these horizons. PSNR is last-frame-at-horizon vs GT (not full prefix).",
	)
	p.add_argument(
		"--val_gt_decode_chunk",
		type=int,
		default=8,
		help="Ignored: AR val GT RGB is decoded only at --val_ar_horizons (one latent frame per horizon).",
	)
	p.add_argument(
		"--cfg_both_drop_prob",
		type=float,
		default=0.10,
		help="Training: fraction of rows with null action and zeroed rule (full CFG dropout).",
	)
	p.add_argument(
		"--cfg_action_drop_prob",
		type=float,
		default=0.05,
		help="Training: null action, keep rule one-hot.",
	)
	p.add_argument(
		"--cfg_rule_drop_prob",
		type=float,
		default=0.05,
		help="Training: zero rule one-hot, keep action.",
	)
	p.add_argument(
		"--cfg_scale_action",
		type=float,
		default=1.5,
		help="Inference / val generate: CFG scale for action (nested with rule).",
	)
	p.add_argument(
		"--cfg_scale_rule",
		type=float,
		default=1.5,
		help="Inference / val generate: CFG scale for rule (nested with action).",
	)
	p.add_argument(
		"--adv_weight",
		type=float,
		default=0.01,
		help="Weight for adversarial rule-invariance loss on state features.",
	)
	p.add_argument(
		"--adv_lambda",
		type=float,
		default=1.0,
		help="Gradient-reversal scale for adversarial rule classifier.",
	)
	p.add_argument(
		"--adv_warmup_steps",
		type=int,
		default=2000,
		help="Linear warmup steps for adversarial term (0 = no warmup).",
	)
	return p.parse_args()


def _parse_int_list(s: str) -> tuple[int, ...]:
	parts = [p.strip() for p in s.split(",") if p.strip()]
	if not parts:
		raise ValueError("expected at least one integer")
	return tuple(int(x) for x in parts)


def _bucket_indices_by_rule(ds_test: MixedEncodedRolloutVideoDataset, canon: tuple[torch.Tensor, ...]) -> list[list[int]]:
	"""Global dataset indices grouped by canonical rule one-hot (order matches ``canon``)."""
	buckets: list[list[int]] = [[] for _ in range(len(canon))]
	for idx in range(len(ds_test)):
		rh = ds_test[idx]["rule_onehot"].float().cpu()
		for ri, c in enumerate(canon):
			if torch.allclose(rh, c, atol=5e-3):
				buckets[ri].append(idx)
				break
	return buckets


def _strip_preview_01(
	chunks: list[torch.Tensor],
	*,
	gap_px: int = 6,
	gap_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
	"""Concatenate ``[1,3,H,W]`` tensors in [0,1] along width → ``[1,3,H,…]`` for TensorBoard.

	Inserts a fixed-width white strip between panels so adjacent frames/rules are easy to tell apart.
	"""
	if not chunks:
		raise ValueError("need at least one image chunk")
	if len(chunks) == 1:
		return chunks[0]
	ref = chunks[0]
	H, device, dtype = int(ref.shape[-2]), ref.device, ref.dtype
	g = max(0, int(gap_px))
	if g == 0:
		return torch.cat(chunks, dim=-1)
	r0, g0, b0 = gap_rgb
	sep = torch.tensor(
		[[[r0]], [[g0]], [[b0]]],
		device=device,
		dtype=dtype,
	).expand(1, 3, H, g)
	parts: list[torch.Tensor] = [chunks[0]]
	for ch in chunks[1:]:
		parts.append(sep)
		parts.append(ch)
	return torch.cat(parts, dim=-1)


def _batched_ranges(n: int, chunk: int) -> list[tuple[int, int]]:
	"""``[(0, c), (c, 2c), …]`` covering ``range(n)`` with ``chunk`` from ``max(1, chunk)``."""
	c = max(1, int(chunk))
	return [(s, min(s + c, n)) for s in range(0, n, c)]


def _base_game_from_encoded_folder(folder: str) -> str:
	"""Strip ``_rules_*`` suffix so variants of one game group together (e.g. ``aliens_rules_fast`` → ``aliens``)."""
	if folder.endswith("_rules_ricochet"):
		return folder[: -len("_rules_ricochet")]
	if folder.endswith("_rules_multishot"):
		return folder[: -len("_rules_multishot")]
	if folder.endswith("_rules_fast"):
		return folder[: -len("_rules_fast")]
	return folder


def _variant_rule_rank(folder: str, base: str) -> tuple[int, str]:
	"""Horizontal order: normal, fast, multishot, ricochet."""
	if folder == base:
		return (0, folder)
	if _base_game_from_encoded_folder(folder) != base:
		return (9, folder)
	if folder.endswith("_rules_fast"):
		return (1, folder)
	if folder.endswith("_rules_multishot"):
		return (2, folder)
	if folder.endswith("_rules_ricochet"):
		return (3, folder)
	return (4, folder)


def _vstack_tgt_gen_rgb01(
	tgt_01: torch.Tensor,
	gen_01: torch.Tensor,
	*,
	gap_px: int = 4,
) -> torch.Tensor:
	"""Target on top, generated below; ``[1,3,H,W]`` in ``[0,1]``."""
	_, _, _, W = tgt_01.shape
	g = max(0, int(gap_px))
	if g == 0:
		return torch.cat([tgt_01, gen_01], dim=-2)
	sep = torch.ones(1, 3, g, W, dtype=tgt_01.dtype, device=tgt_01.device)
	return torch.cat([tgt_01, sep, gen_01], dim=-2)


def _log_val_ar_tb_by_env(
	pred_rgb_h: torch.Tensor,
	tgt_f: torch.Tensor,
	folder_per_row: list[str],
	hid: str,
	writer: SummaryWriter,
	global_step: int,
	*,
	gap_rule_px: int = 6,
	gap_stack_px: int = 4,
) -> None:
	"""One TensorBoard image per base game: columns = rule variants (tgt over gen each), gaps between columns."""
	if not folder_per_row:
		return
	base_to_rows: dict[str, list[int]] = defaultdict(list)
	for i, folder in enumerate(folder_per_row):
		base_to_rows[_base_game_from_encoded_folder(folder)].append(i)
	for base in sorted(base_to_rows.keys()):
		row_inds = sorted(
			base_to_rows[base],
			key=lambda ri: _variant_rule_rank(folder_per_row[ri], base),
		)
		cards: list[torch.Tensor] = []
		for ri in row_inds:
			t_01 = ((tgt_f[ri : ri + 1].clamp(-1, 1) + 1) * 0.5).float()
			g_01 = ((pred_rgb_h[ri : ri + 1].clamp(-1, 1) + 1) * 0.5).float()
			cards.append(_vstack_tgt_gen_rgb01(t_01, g_01, gap_px=gap_stack_px))
		preview = cards[0] if len(cards) == 1 else _strip_preview_01(cards, gap_px=gap_rule_px)
		writer.add_images(f"val/ar_rollout/{hid}/{base}", preview.cpu(), global_step)


def main() -> None:
	args = parse_args()
	if args.env is not None:
		args.env = str(args.env).strip()
		if args.env == "":
			args.env = None
	if args.checkpoint_dir is None:
		args.checkpoint_dir = str(
			DEFAULT_CHECKPOINT_DIR_ALL_ENV_ADV if args.env is None else DEFAULT_CHECKPOINT_DIR_RULES_ADV
		)
	else:
		args.checkpoint_dir = str(Path(args.checkpoint_dir))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	K = args.context_len
	seq_len = K + 1
	encoded_root = Path(args.transitions_root) / args.encoded_subdir

	if args.env is None:
		train_pairs = encoded_dirs_all_under_split(encoded_root / "train")
		test_pairs = encoded_dirs_all_under_split(encoded_root / "test")
	else:
		train_pairs = encoded_dirs_with_rules(encoded_root / "train", args.env)
		test_pairs = encoded_dirs_with_rules(encoded_root / "test", args.env)
	mk_ds = lambda pairs: MixedEncodedRolloutVideoDataset(
		pairs, seq_len=seq_len, stride=1, num_actions=args.num_actions,
	).with_transform(partial(preprocess_latent, history_len=K))
	ds_train = mk_ds(train_pairs)
	ds_test = mk_ds(test_pairs)
	C = int(ds_train[0]["history_latents"].shape[1])
	print(
		f"Dataset windows: train={len(ds_train):,} test={len(ds_test):,}  latent_C={C}  "
		f"rule_sources_train={len(train_pairs)} test={len(test_pairs)}",
	)
	for p, rh in train_pairs:
		print(f"  train shard dir: {p.name}  rule={list(rh)}")

	loader = DataLoader(
		ds_train,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
		persistent_workers=args.num_workers > 0,
	)

	world_model = WorldModel(
		num_actions=args.num_actions,
		cross_attention_dim=CROSS_ATTENTION_DIM,
		vae_checkpoint=args.vae_checkpoint,
		prediction_type=PREDICTION_TYPE,
		history_len=K,
		gradient_checkpointing=args.gradient_checkpointing,
		pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
		cfg_both_drop_prob=args.cfg_both_drop_prob,
		cfg_action_drop_prob=args.cfg_action_drop_prob,
		cfg_rule_drop_prob=args.cfg_rule_drop_prob,
		cfg_scale_action=args.cfg_scale_action,
		cfg_scale_rule=args.cfg_scale_rule,
	).to(device)
	if world_model.latent_channels != C:
		raise ValueError(f"encoded latent C={C} != model latent_channels={world_model.latent_channels}")

	n_diff = sum(p.numel() for p in world_model.diffuser.parameters())
	print(f"Diffuser parameters: {n_diff:,}")

	use_amp = device.type == "cuda" and args.mixed_precision != "no"
	amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
	scaler = GradScaler(enabled=(use_amp and args.mixed_precision == "fp16"))
	optimizer = torch.optim.AdamW(world_model.trainable_parameters(), lr=args.lr, weight_decay=1e-2)

	lpips_val = lpips.LPIPS(net="alex").to(device)
	lpips_val.eval()
	for _p in lpips_val.parameters():
		_p.requires_grad_(False)

	steps_per_epoch = len(loader)
	total_steps = min(args.num_train_epochs * steps_per_epoch, args.max_train_steps)
	scheduler = get_scheduler(
		args.lr_scheduler, optimizer=optimizer,
		num_warmup_steps=args.lr_warmup_steps,
		num_training_steps=total_steps,
	)

	env_tag = "all_envs" if args.env is None else str(args.env)
	run_name = f"{env_tag}_K{K}_encoded_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	writer = SummaryWriter(log_dir=str(Path(args.log_dir) / run_name))
	ckpt_root = Path(args.checkpoint_dir)
	ckpt_root.mkdir(parents=True, exist_ok=True)
	error_buffer = ErrorBuffer(capacity=args.error_buffer_cap)

	val_ar_horizons = _parse_int_list(args.val_ar_horizons)
	if any(h < 1 for h in val_ar_horizons):
		raise ValueError("--val_ar_horizons must be positive integers")
	val_ar_max = max(val_ar_horizons)
	val_ar_horizons_set = frozenset(val_ar_horizons)

	with torch.no_grad():
		canon_rules = canonical_rule_onehots()
		rule_buckets = _bucket_indices_by_rule(ds_test, canon_rules)
		S = int(args.val_samples)
		val_items: list[dict] = []
		val_rule_row_slices: dict[int, tuple[int, int]] = {}
		for ri in range(len(canon_rules)):
			if not rule_buckets[ri]:
				print(f"Warning: no test windows for rule index {ri} ({canon_rules[ri].tolist()}); val skips this rule.")
				continue
			row0 = len(val_items)
			for s in range(S):
				j = rule_buckets[ri][s % len(rule_buckets[ri])]
				val_items.append(ds_test[j])
			val_rule_row_slices[ri] = (row0, len(val_items))
		if not val_items:
			raise RuntimeError("No validation windows: test set empty or no rule buckets matched canonical one-hots.")
		# Keep val caches on CPU (like train_world_model pixel buffers) to free VRAM during training.
		val_hist_z = torch.stack([s["history_latents"] for s in val_items])
		val_tgt_z = torch.stack([s["target_latent"] for s in val_items])
		val_hist_act = torch.stack([s["history_actions"] for s in val_items]).long()
		val_rule_oh = torch.stack([s["rule_onehot"] for s in val_items])

		val_ar_items: list[dict] = []
		val_ar_row_games: list[str] = []
		folder_order = [p.name for p, _ in test_pairs]
		idx_by_folder: defaultdict[str, list[int]] = defaultdict(list)
		for idx in range(len(ds_test)):
			idx_by_folder[ds_test.window_game_folder(idx)].append(idx)
		for folder in folder_order:
			row = None
			for idx in idx_by_folder.get(folder, ()):
				r = ds_test.try_contiguous_ar(idx, K, val_ar_max)
				if r is not None:
					row = r
					break
			if row is None:
				print(
					f"Warning: no AR val window for test folder {folder!r} "
					f"(need {K + val_ar_max} contiguous shard rows); skipping that card.",
				)
				continue
			val_ar_items.append(row)
			val_ar_row_games.append(folder)
		if not val_ar_items:
			print(
				f"Warning: no val windows have K+{val_ar_max} contiguous rows; "
				f"AR val PSNR will be skipped (single-frame val preview only)."
			)
		val_ar_hist_z = torch.stack([s["history_latents"] for s in val_ar_items]) if val_ar_items else None
		val_ar_hist_act = torch.stack([s["history_actions"] for s in val_ar_items]).long() if val_ar_items else None
		val_ar_fut = torch.stack([s["future_action_frames"] for s in val_ar_items]).long() if val_ar_items else None
		val_ar_rule_oh = torch.stack([s["rule_onehot"] for s in val_ar_items]) if val_ar_items else None
		# Decode GT RGB only at horizon end frames (rollout index h-1 for each h in --val_ar_horizons).
		val_ar_gt_rgb_h: dict[int, torch.Tensor] | None = None
		if val_ar_items:
			gt_z = torch.stack([s["gt_future_latents"] for s in val_ar_items])
			vb0 = max(1, int(args.batch_size))
			val_ar_gt_rgb_h = {}
			for h in sorted(val_ar_horizons_set):
				lane = gt_z[:, h - 1 : h].to(device)
				Bar_lane = int(lane.shape[0])
				lane_dec: list[torch.Tensor] = []
				for s0, e0 in _batched_ranges(Bar_lane, vb0):
					lane_dec.append(world_model.decode_video(lane[s0:e0]))
				val_ar_gt_rgb_h[h] = torch.cat(lane_dec, dim=0).squeeze(1).cpu()

	def save_checkpoint(step: int) -> None:
		d = ckpt_root / f"step_{step:07d}"
		world_model.save_diffuser(d)
		torch.save({"step": step, "optimizer": optimizer.state_dict(), "args": vars(args)}, d / "trainer_state.pt")

	global_step = 0
	last_gamma_eff = 0.0
	last_loss: float | None = None
	last_diff_loss: float | None = None
	last_adv_loss: float | None = None
	pbar = tqdm(total=total_steps, desc="Training", unit="step", dynamic_ncols=True)

	while global_step < total_steps:
		for batch in loader:
			if global_step >= total_steps:
				break

			if args.validation_every > 0 and global_step % args.validation_every == 0:
				world_model.eval()
				with torch.no_grad():
					vh = val_hist_z.to(device)
					vt = val_tgt_z.to(device)
					va = val_hist_act.to(device)
					vr = val_rule_oh.to(device)
					Bv = vh.shape[0]
					vb = max(1, int(args.batch_size))
					ts = torch.randint(0, world_model.num_train_timesteps, (Bv,), device=device).long()
					ns = torch.randn_like(vt, dtype=world_model.diffuser.unet.dtype)
					val_mse_num = 0.0
					val_mse_den = 0
					rule_mse_num = {ri: 0.0 for ri in val_rule_row_slices}
					rule_mse_den = {ri: 0 for ri in val_rule_row_slices}
					for s, e in _batched_ranges(Bv, vb):
						pred_b, tgt_b = world_model.diffusion_forward(
							vh[s:e], vt[s:e], va[s:e], ts[s:e], ns[s:e],
							rule_onehot=vr[s:e],
						)
						d = (pred_b.float() - tgt_b.float()).pow(2)
						val_mse_num += d.sum().item()
						val_mse_den += d.numel()
						for ri, (a, b) in val_rule_row_slices.items():
							lo = max(a, s)
							hi = min(b, e)
							if lo < hi:
								dl = d[lo - s : hi - s]
								rule_mse_num[ri] += dl.sum().item()
								rule_mse_den[ri] += dl.numel()
					val_mse = val_mse_num / val_mse_den if val_mse_den else 0.0
					writer.add_scalar("val/mse", val_mse, global_step)
					for ri in val_rule_row_slices:
						if rule_mse_den[ri] > 0:
							writer.add_scalar(f"val/mse_rule_{ri}", rule_mse_num[ri] / rule_mse_den[ri], global_step)
					pbar.set_postfix(
						loss=("—" if last_loss is None else f"{last_loss:.4f}"),
						val_mse=f"{val_mse:.4f}",
						gamma=f"{last_gamma_eff:.4f}",
						buf=len(error_buffer),
					)

					if val_ar_hist_z is not None and val_ar_gt_rgb_h is not None:
						z_ar = val_ar_hist_z.to(device)
						h_act = val_ar_hist_act.to(device)
						fut_dev = val_ar_fut.to(device)
						vr_ar = val_ar_rule_oh.to(device)
						Bar_ar = int(z_ar.shape[0])
						pred_lat_h: dict[int, torch.Tensor] = {}
						for s in tqdm(range(val_ar_max), desc="val AR", leave=False, dynamic_ncols=True):
							fa = h_act[:, -1] if s == 0 else fut_dev[:, s - 1]
							zn_parts: list[torch.Tensor] = []
							for s0, e0 in _batched_ranges(Bar_ar, vb):
								zn_parts.append(
									world_model.generate_next_frame(
										z_ar[s0:e0], h_act[s0:e0], fa[s0:e0],
										num_inference_steps=int(args.num_inference_steps),
										rule_onehot=vr_ar[s0:e0],
									)
								)
							z_next = torch.cat(zn_parts, dim=0)
							hz = s + 1
							if hz in val_ar_horizons_set:
								pred_lat_h[hz] = z_next.detach()
							z_ar = torch.cat([z_ar[:, 1:], z_next], dim=1)
							h_act = torch.cat([h_act[:, 1:], fa.unsqueeze(1)], dim=1)
						hs_unique = sorted(val_ar_horizons_set)
						lat_pack = torch.cat([pred_lat_h[h] for h in hs_unique], dim=1)
						del pred_lat_h
						rgb_pack_parts: list[torch.Tensor] = []
						for s0, e0 in _batched_ranges(Bar_ar, vb):
							rgb_pack_parts.append(world_model.decode_video(lat_pack[s0:e0]))
						rgb_pack = torch.cat(rgb_pack_parts, dim=0)
						del lat_pack
						h_to_idx = {h: i for i, h in enumerate(hs_unique)}
						k_h = len(hs_unique)
						lp_p = torch.cat([rgb_pack[:, i].float().clamp(-1, 1) for i in range(k_h)], dim=0)
						lp_t = torch.cat(
							[val_ar_gt_rgb_h[h].to(device).float().clamp(-1, 1) for h in hs_unique],
							dim=0,
						)
						lp_raw = lpips_val(lp_p, lp_t).flatten().reshape(k_h, Bar_ar).mean(dim=1)
						lp_per_h = {hs_unique[i]: float(lp_raw[i].item()) for i in range(k_h)}
						del lp_p, lp_t, lp_raw
						for h in val_ar_horizons:
							idx = h_to_idx[h]
							pred_rgb_h = rgb_pack[:, idx]
							tgt_f = val_ar_gt_rgb_h[h].to(device)
							hid = f"h{h:02d}"
							writer.add_scalar(
								f"val/psnr_ar/{hid}",
								psnr_neg1_to_01(pred_rgb_h.unsqueeze(1), tgt_f.unsqueeze(1)),
								global_step,
							)
							writer.add_scalar(f"val/lpips_ar/{hid}", lp_per_h[h], global_step)
							_log_val_ar_tb_by_env(
								pred_rgb_h, tgt_f, val_ar_row_games, hid, writer, global_step,
							)
							del tgt_f
						del rgb_pack
					else:
						cl_parts: list[torch.Tensor] = []
						for s, e in _batched_ranges(Bv, vb):
							cl_parts.append(
								world_model.generate_next_frame(
									vh[s:e], va[s:e], va[s:e, -1],
									num_inference_steps=int(args.num_inference_steps),
									rule_onehot=vr[s:e],
								)
							)
						chunk_lat = torch.cat(cl_parts, dim=0)
						dec_parts: list[torch.Tensor] = []
						for s, e in _batched_ranges(Bv, vb):
							dec_parts.append(world_model.decode_video(chunk_lat[s:e]))
						dec1 = torch.cat(dec_parts, dim=0)
						vt_rgb_parts: list[torch.Tensor] = []
						for s, e in _batched_ranges(Bv, vb):
							vt_rgb_parts.append(world_model.decode_frames(vt[s:e]))
						vt_rgb = torch.cat(vt_rgb_parts, dim=0)
						strip_idx = [val_rule_row_slices[ri][0] for ri in sorted(val_rule_row_slices.keys())]
						gen_parts = [((dec1[i : i + 1, 0].clamp(-1, 1) + 1) * 0.5) for i in strip_idx]
						tgt_parts = [((vt_rgb[i : i + 1].clamp(-1, 1) + 1) * 0.5) for i in strip_idx]
						writer.add_images(
							"val/generated_f0_by_rule",
							_strip_preview_01([g.cpu() for g in gen_parts]),
							global_step,
						)
						writer.add_images(
							"val/target_f0_by_rule",
							_strip_preview_01([t.cpu() for t in tgt_parts]),
							global_step,
						)
					del vh, vt, va, vr
				world_model.train()

			optimizer.zero_grad(set_to_none=True)
			z_hist = batch["history_latents"].to(device)
			z_tgt = batch["target_latent"].to(device)
			hist_actions = batch["history_actions"].to(device)
			rule_oh = batch["rule_onehot"].to(device)
			B = z_hist.shape[0]

			Wg = args.gamma_warmup_steps
			gamma_eff = float(args.gamma) if Wg <= 0 else float(args.gamma) * min(1.0, global_step / float(Wg))
			last_gamma_eff = gamma_eff
			Wa = args.adv_warmup_steps
			adv_scale = 1.0 if Wa <= 0 else min(1.0, global_step / float(Wa))
			adv_weight_eff = float(args.adv_weight) * adv_scale
			adv_lambda_eff = float(args.adv_lambda) * adv_scale

			delta_hist = error_buffer.sample_like(z_hist) if error_buffer.ready() else None
			timesteps = torch.randint(0, world_model.num_train_timesteps, (B,), device=device).long()
			noise = torch.randn_like(z_tgt, dtype=world_model.diffuser.unet.dtype)
			rule_ids = rule_oh.argmax(dim=1)

			with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
				model_pred, target, h_state = world_model.diffusion_forward(
					z_hist, z_tgt, hist_actions, timesteps, noise,
					delta_hist=delta_hist, gamma=gamma_eff,
					rule_onehot=rule_oh,
					return_state=True,
				)
				diff_loss = F.mse_loss(model_pred.float(), target.float())
				# Adversary uses the same state feature that conditions diffusion.
				adv_logits = world_model.adversarial_rule_logits_from_state(h_state, adv_lambda=adv_lambda_eff)
				adv_loss = F.cross_entropy(adv_logits.float(), rule_ids)
				loss = diff_loss + adv_weight_eff * adv_loss
			last_loss = loss.item()
			last_diff_loss = diff_loss.item()
			last_adv_loss = adv_loss.item()

			if scaler.is_enabled():
				scaler.scale(loss).backward()
			else:
				loss.backward()

			with torch.no_grad():
				alpha_bar = world_model.diffuser.noise_scheduler.alphas_cumprod.to(device)[timesteps]
				sqrt_a = alpha_bar.sqrt().view(B, 1, 1, 1)
				sqrt_1ma = (1 - alpha_bar).sqrt().view(B, 1, 1, 1)
				noisy_tgt = (sqrt_a * z_tgt + sqrt_1ma * noise).to(model_pred.dtype)
				pt = world_model.diffuser.noise_scheduler.config.prediction_type
				if pt == "v_prediction":
					z_hat = sqrt_a * noisy_tgt - sqrt_1ma * model_pred
				elif pt == "sample":
					z_hat = model_pred
				else:
					z_hat = (noisy_tgt - sqrt_1ma * model_pred) / sqrt_a.clamp(min=1e-8)
				delta_fut = z_hat - z_tgt.to(z_hat.dtype)
				error_buffer.push(future_residuals_as_history_block(delta_fut, K))

			if scaler.is_enabled():
				scaler.unscale_(optimizer)
			if args.max_grad_norm > 0:
				torch.nn.utils.clip_grad_norm_(world_model.trainable_parameters(), args.max_grad_norm)
			if scaler.is_enabled():
				scaler.step(optimizer)
				scaler.update()
			else:
				optimizer.step()
			scheduler.step()
			global_step += 1
			pbar.update(1)
			pbar.set_postfix(
				loss=f"{loss.item():.4f}",
				diff=f"{(last_diff_loss if last_diff_loss is not None else float('nan')):.4f}",
				adv=f"{(last_adv_loss if last_adv_loss is not None else float('nan')):.4f}",
				gamma=f"{last_gamma_eff:.4f}",
				buf=len(error_buffer),
			)

			if global_step > 0 and global_step % 20 == 0:
				writer.add_scalar("train/loss", loss.item(), global_step)
				writer.add_scalar("train/diff_loss", diff_loss.item(), global_step)
				writer.add_scalar("train/adv_loss", adv_loss.item(), global_step)
				writer.add_scalar("train/adv_weight_eff", adv_weight_eff, global_step)
				writer.add_scalar("train/adv_lambda_eff", adv_lambda_eff, global_step)
				writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

			if global_step > 0 and args.save_every > 0 and global_step % args.save_every == 0:
				save_checkpoint(global_step)

	save_checkpoint(global_step)
	writer.close()
	pbar.close()
	print("Training finished.")


if __name__ == "__main__":
	main()
