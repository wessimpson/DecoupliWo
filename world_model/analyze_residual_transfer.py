"""Analyze whether residual corrections improve variant transfer and cluster by rule."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from world_model.dataset import (
	MixedEncodedRolloutVideoDataset,
	base_game_from_encoded_folder,
	encoded_variant_dirs_under_split,
	preprocess_latent,
)
from world_model.model.checkpoint import load_residual_world_model


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--base_ckpt_dir", type=str, required=True)
	p.add_argument("--residual_ckpt_dir", type=str, required=True)
	p.add_argument("--transitions_root", type=str, default=str(Path("data") / "transitions"))
	p.add_argument("--encoded_subdir", type=str, default="encoded")
	p.add_argument("--split", type=str, choices=["train", "test"], default="test")
	p.add_argument("--env", type=str, default=None, help="Base game or explicit variant. Omit for all variants.")
	p.add_argument("--vae_checkpoint", type=str, default=str(Path("world_model") / "checkpoints" / "vae" / "vae.pt"))
	p.add_argument("--num_actions", type=int, default=7)
	p.add_argument("--context_len", type=int, default=4)
	p.add_argument("--batch_size", type=int, default=8)
	p.add_argument("--num_workers", type=int, default=4)
	p.add_argument("--max_samples", type=int, default=512)
	p.add_argument("--output_dir", type=str, default=str(Path("analysis") / "residual_transfer"))
	return p.parse_args()


def _pairwise_centroid_distance(centroids: list[torch.Tensor]) -> float:
	if len(centroids) < 2:
		return 0.0
	vals = []
	for i in range(len(centroids)):
		for j in range(i + 1, len(centroids)):
			vals.append(torch.linalg.vector_norm(centroids[i] - centroids[j]).item())
	return float(sum(vals) / max(1, len(vals)))


def _centroid_stats(features: torch.Tensor, labels: list[str]) -> tuple[float, float, int]:
	buckets: dict[str, list[int]] = defaultdict(list)
	for i, lab in enumerate(labels):
		buckets[str(lab)].append(i)
	centroids = []
	within = []
	for inds in buckets.values():
		x = features[inds]
		c = x.mean(dim=0)
		centroids.append(c)
		within.append(torch.linalg.vector_norm(x - c, dim=1).mean().item())
	return _pairwise_centroid_distance(centroids), float(sum(within) / max(1, len(within))), len(buckets)


def _rule_label(rule: torch.Tensor) -> str:
	names = ("fast", "multishot", "ricochet")
	active = [names[i] for i, v in enumerate(rule.tolist()) if float(v) > 0.5]
	return "+".join(active) if active else "normal"


def main() -> None:
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	K = int(args.context_len)
	seq_len = K + 1
	encoded_root = Path(args.transitions_root) / args.encoded_subdir / args.split
	pairs = encoded_variant_dirs_under_split(encoded_root, env=args.env)
	ds = MixedEncodedRolloutVideoDataset(
		pairs, seq_len=seq_len, stride=1, num_actions=args.num_actions,
	).with_transform(partial(preprocess_latent, history_len=K))
	n = min(int(args.max_samples), len(ds))
	if n <= 0:
		raise RuntimeError("empty analysis dataset")
	subset = Subset(ds, list(range(n)))
	loader = DataLoader(
		subset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
	)
	model = load_residual_world_model(
		args.base_ckpt_dir,
		args.residual_ckpt_dir,
		num_actions=args.num_actions,
		history_len=K,
		vae_checkpoint=args.vae_checkpoint,
		device=device,
	)
	model.eval()

	base_mse_num = 0.0
	combined_mse_num = 0.0
	residual_mse_num = 0.0
	den = 0
	features = []
	rule_labels = []
	game_labels = []
	row0 = 0
	with torch.no_grad():
		for batch in tqdm(loader, desc="analyze", dynamic_ncols=True):
			z_hist = batch["history_latents"].to(device)
			z_tgt = batch["target_latent"].to(device)
			hist_actions = batch["history_actions"].to(device)
			rule_oh = batch["rule_onehot"].to(device).float()
			B = int(z_hist.shape[0])
			timesteps = torch.randint(0, model.num_train_timesteps, (B,), device=device).long()
			noise = torch.randn_like(z_tgt, dtype=model.residual_model.diffuser.unet.dtype)
			delta_pred, delta_target, base_pred, full_target = model.residual_forward(
				z_hist, z_tgt, hist_actions, timesteps, noise, rule_oh,
			)
			combined = base_pred + delta_pred
			base_mse_num += F.mse_loss(base_pred.float(), full_target.float(), reduction="sum").item()
			combined_mse_num += F.mse_loss(combined.float(), full_target.float(), reduction="sum").item()
			residual_mse_num += F.mse_loss(delta_pred.float(), delta_target.float(), reduction="sum").item()
			den += full_target.numel()
			features.append(delta_pred.float().mean(dim=(2, 3)).cpu())
			for i in range(B):
				row = ds[row0 + i]
				folder = ds.window_game_folder(row0 + i)
				game_labels.append(base_game_from_encoded_folder(folder))
				rule_labels.append(_rule_label(row["rule_onehot"]))
			row0 += B

	feat = torch.cat(features, dim=0)
	rule_between, rule_within, n_rules = _centroid_stats(feat, rule_labels)
	game_between, game_within, n_games = _centroid_stats(feat, game_labels)
	base_mse = base_mse_num / max(1, den)
	combined_mse = combined_mse_num / max(1, den)
	residual_mse = residual_mse_num / max(1, den)
	improvement = 0.0 if base_mse <= 0 else 100.0 * (base_mse - combined_mse) / base_mse

	out_dir = Path(args.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	report = out_dir / "report.md"
	report.write_text(
		"\n".join([
			"# Residual Transfer Analysis",
			"",
			f"- Split: `{args.split}`",
			f"- Env filter: `{args.env or 'all variants'}`",
			f"- Samples: `{n}`",
			f"- Base diffusion MSE: `{base_mse:.6g}`",
			f"- Base + residual diffusion MSE: `{combined_mse:.6g}`",
			f"- Relative MSE improvement: `{improvement:.2f}%`",
			f"- Residual target MSE: `{residual_mse:.6g}`",
			"",
			"## Residual Feature Separation",
			"",
			f"- Rule groups: `{n_rules}`, centroid distance: `{rule_between:.6g}`, within distance: `{rule_within:.6g}`",
			f"- Game groups: `{n_games}`, centroid distance: `{game_between:.6g}`, within distance: `{game_within:.6g}`",
			f"- Rule/game separation ratio: `{(rule_between / max(game_between, 1e-12)):.6g}`",
			"",
			"A higher rule/game ratio is evidence that the correction output is organized more by rule than by game visuals.",
		]),
		encoding="utf-8",
	)
	print(f"Wrote {report.resolve()}")


if __name__ == "__main__":
	main()
