"""Validation for ``train_dynamics``: one fixed seeded draw of test env folders + windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from world_model.dataset import RULE_TAGS, encoded_folder_base_game
from world_model.util.evaluation import psnr as psnr_neg1_to_01
from world_model.util.rules import inject_rule_tag, rule_onehot_from_tag_list
from world_model.util.visualization import batched_ranges, neg1_to_01, strip_horizontal, vstack_tgt_gen

COUNTERFACTUAL_STEPS = 10
COUNTERFACTUAL_SHOOT_ACTION = 5
COUNTERFACTUAL_SPECS: tuple[tuple[str, str], ...] = (
	("defender", "multishot_5"),
	("defender", "shoot_walls"),
	("roadfighter", "multishot_5"),
	("roadfighter", "shoot_walls"),
	("jaws", "multishot_5"),
	("jaws", "shoot_walls"),
)

RULE_COMPOSITION_ENV = "aliens"
RULE_COMPOSITION_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
	("big_shot_4x+multishot_5", ("big_shot_4x", "multishot_5")),
	("two_hit_color+multishot_5", ("two_hit_color", "multishot_5")),
	("ricochet+multishot_5", ("ricochet", "multishot_5")),
	("enemy_explode+multishot_5", ("enemy_explode", "multishot_5")),
	("big_shot_4x+shoot_walls", ("big_shot_4x", "shoot_walls")),
	("big_shot_4x+enemy_explode+multishot_5", ("big_shot_4x", "enemy_explode", "multishot_5")),
)

if TYPE_CHECKING:
	from collections import defaultdict

	from world_model.dataset import MixedEncodedRolloutVideoDataset
	from world_model.model.world_model import WorldModel


def first_latent_chw(env_dir: Path) -> tuple[int, int, int]:
	for shard in sorted(env_dir.glob("shard_*")):
		lat = shard / "latent.npy"
		if lat.is_file():
			arr = torch.from_numpy(np.load(lat, mmap_mode="r"))
			if arr.ndim != 4:
				raise ValueError(f"Unexpected latent shape in {lat}: {tuple(arr.shape)}")
			return int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[3])
	raise FileNotFoundError(f"No shard_*/latent.npy under {env_dir}")


def _indices_by_folder(ds: MixedEncodedRolloutVideoDataset) -> dict[str, list[int]]:
	from collections import defaultdict

	out: defaultdict[str, list[int]] = defaultdict(list)
	for idx in range(len(ds)):
		out[ds.window_game_folder(idx)].append(idx)
	return dict(out)


def _pick_ar_window(
	ds: MixedEncodedRolloutVideoDataset,
	folder: str,
	indices: list[int],
	*,
	K: int,
	val_ar_max: int,
	rng: np.random.Generator,
) -> dict[str, torch.Tensor] | None:
	if not indices:
		return None
	for idx in rng.permutation(indices):
		row = ds.try_contiguous_ar(idx, K, val_ar_max)
		if row is not None:
			return row
	return None


@dataclass
class DynamicsValPack:
	val_hist_z: torch.Tensor
	val_tgt_z: torch.Tensor
	val_hist_act: torch.Tensor
	val_rule_oh: torch.Tensor
	val_ar_hist_z: torch.Tensor
	val_ar_hist_act: torch.Tensor
	val_ar_fut: torch.Tensor
	val_ar_rule_oh: torch.Tensor
	val_ar_gt_rgb_h: dict[int, torch.Tensor]
	folder_names: list[str]
	counterfactual_rows: dict[str, dict]
	rule_composition_row: dict | None


def build_val_pack(
	ds_test: MixedEncodedRolloutVideoDataset,
	test_pairs: list[tuple[Path, tuple[float, ...]]],
	*,
	K: int,
	val_samples: int,
	val_ar_max: int,
	val_ar_horizons_set: frozenset[int],
	val_seed: int,
	batch_size: int,
	device: torch.device,
	world_model: WorldModel,
) -> DynamicsValPack:
	"""Sample ``val_samples`` test env folders (no replacement) and one random AR-valid window each."""
	rng = np.random.default_rng(int(val_seed))
	folders = [p.name for p, _ in test_pairs]
	if not folders:
		raise RuntimeError("No test env folders for validation.")
	n_pick = min(int(val_samples), len(folders))
	picked = list(rng.choice(folders, size=n_pick, replace=False))
	idx_by_folder = _indices_by_folder(ds_test)

	rows: list[dict] = []
	folder_names: list[str] = []
	for folder in picked:
		row = _pick_ar_window(ds_test, folder, idx_by_folder.get(folder, []), K=K, val_ar_max=val_ar_max, rng=rng)
		if row is None:
			print(f"Warning: no AR-valid window in test folder {folder!r} (need {K + val_ar_max} contiguous rows); skipped.")
			continue
		rows.append(row)
		folder_names.append(folder)
	if not rows:
		raise RuntimeError("No validation windows: no sampled folder had a long enough contiguous segment.")

	print(f"Val set ({len(rows)}): seed={val_seed} folders={folder_names}")

	val_hist_z = torch.stack([r["history_latents"] for r in rows])
	val_tgt_z = torch.stack([r["gt_future_latents"][0] for r in rows])  # [C,h,w] not [:,0] (channel slice)
	val_hist_act = torch.stack([r["history_actions"] for r in rows]).long()
	val_rule_oh = torch.stack([r["rule_onehot"] for r in rows])
	val_ar_hist_z = val_hist_z
	val_ar_hist_act = val_hist_act
	val_ar_fut = torch.stack([r["future_action_frames"] for r in rows]).long()
	val_ar_rule_oh = val_rule_oh

	gt_z = torch.stack([r["gt_future_latents"] for r in rows])
	vb0 = max(1, int(batch_size))
	val_ar_gt_rgb_h: dict[int, torch.Tensor] = {}
	for h in sorted(val_ar_horizons_set):
		lane = gt_z[:, h - 1 : h].to(device)
		val_ar_gt_rgb_h[h] = torch.cat(
			[world_model.decode_video(lane[s0:e0]) for s0, e0 in batched_ranges(lane.shape[0], vb0)],
			dim=0,
		).squeeze(1).cpu()

	counterfactual_rows: dict[str, dict] = {}
	cf_bases = {base for base, _ in COUNTERFACTUAL_SPECS}
	for base in sorted(cf_bases):
		cf_indices = idx_by_folder.get(base, [])
		if not cf_indices:
			cf_indices = [
				i for folder, inds in idx_by_folder.items()
				if encoded_folder_base_game(folder) == base
				for i in inds
			]
		row = _pick_ar_window(ds_test, base, cf_indices, K=K, val_ar_max=COUNTERFACTUAL_STEPS, rng=rng)
		if row is None:
			print(
				f"Warning: no counter-factual seed for {base!r} "
				f"(need {K + COUNTERFACTUAL_STEPS} contiguous rows under test/{base} or variants).",
			)
		else:
			counterfactual_rows[base] = row

	rng_comp = np.random.default_rng(int(val_seed) + 911)
	comp_indices = idx_by_folder.get(RULE_COMPOSITION_ENV, [])
	if not comp_indices:
		comp_indices = [
			i for folder, inds in idx_by_folder.items()
			if encoded_folder_base_game(folder) == RULE_COMPOSITION_ENV
			for i in inds
		]
	rule_composition_row = _pick_ar_window(
		ds_test, RULE_COMPOSITION_ENV, comp_indices,
		K=K, val_ar_max=COUNTERFACTUAL_STEPS, rng=rng_comp,
	)
	if rule_composition_row is None:
		print(
			f"Warning: no rule-composition seed for {RULE_COMPOSITION_ENV!r} "
			f"(need {K + COUNTERFACTUAL_STEPS} contiguous rows).",
		)

	return DynamicsValPack(
		val_hist_z=val_hist_z,
		val_tgt_z=val_tgt_z,
		val_hist_act=val_hist_act,
		val_rule_oh=val_rule_oh,
		val_ar_hist_z=val_ar_hist_z,
		val_ar_hist_act=val_ar_hist_act,
		val_ar_fut=val_ar_fut,
		val_ar_rule_oh=val_ar_rule_oh,
		val_ar_gt_rgb_h=val_ar_gt_rgb_h,
		folder_names=folder_names,
		counterfactual_rows=counterfactual_rows,
		rule_composition_row=rule_composition_row,
	)


def _log_ar_rollout_strip(
	pred_rgb_h: torch.Tensor,
	tgt_f: torch.Tensor,
	folder_names: list[str],
	hid: str,
	writer: SummaryWriter,
	global_step: int,
) -> None:
	cards = [
		vstack_tgt_gen(neg1_to_01(tgt_f[i : i + 1]), neg1_to_01(pred_rgb_h[i : i + 1]))
		for i in range(len(folder_names))
	]
	preview = cards[0] if len(cards) == 1 else strip_horizontal(cards)
	writer.add_images(f"val/ar_rollout/{hid}", preview.cpu(), global_step)
	for i, name in enumerate(folder_names):
		writer.add_images(
			f"val/ar_rollout/{hid}/{name}",
			vstack_tgt_gen(neg1_to_01(tgt_f[i : i + 1]), neg1_to_01(pred_rgb_h[i : i + 1])).cpu(),
			global_step,
		)


def _rollout_shoot_strip(
	world_model: WorldModel,
	row: dict,
	rule_oh: torch.Tensor,
	*,
	device: torch.device,
	num_inference_steps: int,
	num_actions: int,
	steps: int,
) -> torch.Tensor:
	shoot = min(COUNTERFACTUAL_SHOOT_ACTION, num_actions - 1)
	z = row["history_latents"].unsqueeze(0).to(device)
	h_act = row["history_actions"].unsqueeze(0).long().to(device)
	frames_01: list[torch.Tensor] = []
	for _ in range(steps):
		fa = torch.full((1,), shoot, device=device, dtype=torch.long)
		z_next = world_model.generate_next_frame(
			z, h_act, fa, num_inference_steps=num_inference_steps, rule_onehot=rule_oh,
		)
		dec = world_model.decode_video(z_next)[0, 0]
		frames_01.append(neg1_to_01(dec.unsqueeze(0)).cpu())
		z = torch.cat([z[:, 1:], z_next], dim=1)
		h_act = torch.cat([h_act[:, 1:], fa.unsqueeze(1)], dim=1)
	return strip_horizontal(frames_01)


def _run_counterfactual_rollouts(
	world_model: WorldModel,
	writer: SummaryWriter,
	global_step: int,
	pack: DynamicsValPack,
	*,
	device: torch.device,
	num_inference_steps: int,
	num_actions: int,
) -> None:
	if not pack.counterfactual_rows:
		return
	dtype = pack.val_rule_oh.dtype
	for base, rule_tag in COUNTERFACTUAL_SPECS:
		row = pack.counterfactual_rows.get(base)
		if row is None:
			continue
		if rule_tag not in RULE_TAGS:
			print(f"Warning: counter-factual tag {rule_tag!r} not in RULE_TAGS; skip {base}+{rule_tag}.")
			continue
		vr_cf = inject_rule_tag(torch.zeros(1, pack.val_rule_oh.shape[1], device=device, dtype=dtype), rule_tag)
		writer.add_images(
			f"counter-factual/{base}+{rule_tag}",
			_rollout_shoot_strip(
				world_model, row, vr_cf, device=device,
				num_inference_steps=num_inference_steps, num_actions=num_actions,
				steps=COUNTERFACTUAL_STEPS,
			),
			global_step,
		)


def _run_rule_composition_rollouts(
	world_model: WorldModel,
	writer: SummaryWriter,
	global_step: int,
	pack: DynamicsValPack,
	*,
	device: torch.device,
	num_inference_steps: int,
	num_actions: int,
) -> None:
	row = pack.rule_composition_row
	if row is None:
		return
	for label, tags in RULE_COMPOSITION_SPECS:
		try:
			vr = rule_onehot_from_tag_list(tags, device=device)
		except KeyError as e:
			print(f"Warning: rule-composition {label!r} skipped: {e}")
			continue
		writer.add_images(
			f"rule_composition/{label}",
			_rollout_shoot_strip(
				world_model, row, vr, device=device,
				num_inference_steps=num_inference_steps, num_actions=num_actions,
				steps=COUNTERFACTUAL_STEPS,
			),
			global_step,
		)


def run_dynamics_validation(
	world_model: WorldModel,
	writer: SummaryWriter,
	global_step: int,
	pack: DynamicsValPack,
	*,
	device: torch.device,
	batch_size: int,
	num_inference_steps: int,
	num_actions: int,
	val_ar_horizons: tuple[int, ...],
	val_ar_horizons_set: frozenset[int],
	val_ar_max: int,
	lpips_val: torch.nn.Module,
	pbar: tqdm | None = None,
) -> float:
	"""MSE, AR PSNR/LPIPS, and rollout images on the same ``val_samples`` windows."""
	world_model.eval()
	vh = pack.val_hist_z.to(device)
	vt = pack.val_tgt_z.to(device)
	va = pack.val_hist_act.to(device)
	vr = pack.val_rule_oh.to(device)
	Bv, vb = vh.shape[0], max(1, int(batch_size))

	ts = torch.randint(0, world_model.num_train_timesteps, (Bv,), device=device).long()
	ns = torch.randn_like(vt, dtype=world_model.diffuser.unet.dtype)
	val_mse_num = val_mse_den = 0.0
	for s, e in batched_ranges(Bv, vb):
		pred_b, tgt_b = world_model.diffusion_forward(
			vh[s:e], vt[s:e], va[s:e], ts[s:e], ns[s:e], rule_onehot=vr[s:e],
		)
		d = (pred_b.float() - tgt_b.float()).pow(2)
		val_mse_num += d.sum().item()
		val_mse_den += d.numel()
	val_mse = val_mse_num / val_mse_den if val_mse_den else 0.0
	writer.add_scalar("val/mse", val_mse, global_step)
	if pbar is not None:
		pbar.set_postfix(val_mse=f"{val_mse:.4f}")

	z_ar = pack.val_ar_hist_z.to(device)
	h_act = pack.val_ar_hist_act.to(device)
	fut_dev = pack.val_ar_fut.to(device)
	vr_ar = pack.val_ar_rule_oh.to(device)
	Bar = int(z_ar.shape[0])
	pred_lat_h: dict[int, torch.Tensor] = {}
	for s in tqdm(range(val_ar_max), desc="val AR", leave=False, dynamic_ncols=True):
		fa = h_act[:, -1] if s == 0 else fut_dev[:, s - 1]
		zn = torch.cat([
			world_model.generate_next_frame(
				z_ar[s0:e0], h_act[s0:e0], fa[s0:e0],
				num_inference_steps=num_inference_steps,
				rule_onehot=vr_ar[s0:e0],
			)
			for s0, e0 in batched_ranges(Bar, vb)
		], dim=0)
		hz = s + 1
		if hz in val_ar_horizons_set:
			pred_lat_h[hz] = zn.detach()
		z_ar = torch.cat([z_ar[:, 1:], zn], dim=1)
		h_act = torch.cat([h_act[:, 1:], fa.unsqueeze(1)], dim=1)

	hs_unique = sorted(val_ar_horizons_set)
	lat_pack = torch.cat([pred_lat_h[h] for h in hs_unique], dim=1)
	rgb_pack = torch.cat([
		world_model.decode_video(lat_pack[s0:e0])
		for s0, e0 in batched_ranges(Bar, vb)
	], dim=0)
	h_to_idx = {h: i for i, h in enumerate(hs_unique)}
	k_h = len(hs_unique)
	lp_p = torch.cat([rgb_pack[:, i].float().clamp(-1, 1) for i in range(k_h)], dim=0)
	lp_t = torch.cat([pack.val_ar_gt_rgb_h[h].to(device).float().clamp(-1, 1) for h in hs_unique], dim=0)
	lp_raw = lpips_val(lp_p, lp_t).flatten().reshape(k_h, Bar).mean(dim=1)
	lp_per_h = {hs_unique[i]: float(lp_raw[i].item()) for i in range(k_h)}

	for h in val_ar_horizons:
		idx = h_to_idx[h]
		pred_rgb_h = rgb_pack[:, idx]
		tgt_f = pack.val_ar_gt_rgb_h[h].to(device)
		hid = f"h{h:02d}"
		writer.add_scalar(f"val/psnr_ar/{hid}", psnr_neg1_to_01(pred_rgb_h.unsqueeze(1), tgt_f.unsqueeze(1)), global_step)
		writer.add_scalar(f"val/lpips_ar/{hid}", lp_per_h[h], global_step)
		_log_ar_rollout_strip(pred_rgb_h, tgt_f, pack.folder_names, hid, writer, global_step)

	# Same windows: one-step generate vs GT (aligned with AR step 0)
	cl_parts = [
		world_model.generate_next_frame(
			vh[s:e], va[s:e], va[s:e, -1],
			num_inference_steps=num_inference_steps,
			rule_onehot=vr[s:e],
		)
		for s, e in batched_ranges(Bv, vb)
	]
	chunk_lat = torch.cat(cl_parts, dim=0)
	dec1 = torch.cat([world_model.decode_video(chunk_lat[s:e]) for s, e in batched_ranges(Bv, vb)], dim=0)
	vt_rgb = torch.cat([world_model.decode_frames(vt[s:e]) for s, e in batched_ranges(Bv, vb)], dim=0)
	gen_cards = [neg1_to_01(dec1[i : i + 1, 0]) for i in range(Bv)]
	tgt_cards = [neg1_to_01(vt_rgb[i : i + 1]) for i in range(Bv)]
	writer.add_images("val/one_step_strip", strip_horizontal(gen_cards).cpu(), global_step)
	writer.add_images("val/one_step_tgt_strip", strip_horizontal(tgt_cards).cpu(), global_step)

	_run_counterfactual_rollouts(
		world_model, writer, global_step, pack,
		device=device, num_inference_steps=num_inference_steps, num_actions=num_actions,
	)
	_run_rule_composition_rollouts(
		world_model, writer, global_step, pack,
		device=device, num_inference_steps=num_inference_steps, num_actions=num_actions,
	)

	return val_mse
