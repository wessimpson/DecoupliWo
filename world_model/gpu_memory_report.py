"""
GPU memory report: driver totals, PyTorch caching allocator, and optional model breakdown.

Run from repo root:
  python -m world_model.gpu_memory_report              # driver + empty PyTorch stats
  python -m world_model.gpu_memory_report --world-model --trainable_parts motion_modules --forward
  python -m world_model.gpu_memory_report --world-model --trainable_parts lora_attn --lora_rank 8 --forward --backward
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast

from world_model.model.net.trainable_parts import TRAINABLE_PARTS_CHOICES, count_trainable_params
from world_model.model.world_model import WorldModel


def _fmt_bytes(n: int) -> str:
	if n < 1024:
		return f"{n} B"
	for unit in ("KiB", "MiB", "GiB", "TiB"):
		n /= 1024.0
		if n < 1024.0:
			return f"{n:.2f} {unit}"
	return f"{n:.2f} PiB"


def _device_index(device: torch.device) -> int:
	if device.type != "cuda":
		raise SystemExit("This script needs CUDA (torch.device('cuda'))")
	return device.index if device.index is not None else 0


def print_driver_memory(device: torch.device) -> None:
	idx = _device_index(device)
	free_b, total_b = torch.cuda.mem_get_info(idx)
	used_b = total_b - free_b
	name = torch.cuda.get_device_name(idx)
	print(f"\n=== Driver ({name}, cuda:{idx}) ===")
	print(f"  Total VRAM:     {_fmt_bytes(total_b)}")
	print(f"  Used (all procs): {_fmt_bytes(used_b)}")
	print(f"  Free:           {_fmt_bytes(free_b)}")


def print_driver_vs_pytorch_scope() -> None:
	print(
		"\n=== Driver vs PyTorch (why numbers differ) ===\n"
		"  Driver 'Used (all procs)': VRAM in use on the GPU by *all* processes - other apps,\n"
		"  other Python runs, the compositor, and any CUDA context already resident - not only this script.\n"
		"  PyTorch stats below: *this Python process* only. They read 0 B until this process allocates\n"
		"  tensors on CUDA (for example after --world-model)."
	)


def print_allocator_legend(device: torch.device) -> None:
	"""PyTorch caching allocator: what the numbers mean."""
	idx = _device_index(device)
	s = torch.cuda.memory_stats(idx)
	alloc = torch.cuda.memory_allocated(idx)
	reserved = torch.cuda.memory_reserved(idx)
	peak_alloc = torch.cuda.max_memory_allocated(idx)
	peak_reserved = torch.cuda.max_memory_reserved(idx)
	active = int(s.get("active_bytes.all.current", 0))
	inactive = int(s.get("inactive_split_bytes.all.current", 0))

	print("\n=== PyTorch CUDA allocator (this process) ===")
	print(
		"  allocated:  memory for live tensors the program is using\n"
		"  reserved:   blocks grabbed from the driver (allocated + free pools inside PyTorch)\n"
		"  active:     bytes inside allocated tensors\n"
		"  inactive:   reserved but not tied to a current tensor (fragmentation / pool)"
	)
	print(f"  allocated (current): {_fmt_bytes(alloc)}")
	print(f"  allocated (peak):    {_fmt_bytes(peak_alloc)}")
	print(f"  reserved (current):  {_fmt_bytes(reserved)}")
	print(f"  reserved (peak):     {_fmt_bytes(peak_reserved)}")
	print(f"  active (current):    {_fmt_bytes(active)}")
	print(f"  inactive split:      {_fmt_bytes(inactive)}")
	if reserved > 0:
		pct = 100.0 * alloc / reserved
		print(f"  utilization (allocated/reserved): {pct:.1f}%")


def print_torch_memory_summary(device: torch.device) -> None:
	idx = _device_index(device)
	print("\n=== torch.cuda.memory_summary (detail) ===")
	print(torch.cuda.memory_summary(device=idx, abbreviated=False))


def _top_level_prefix(name: str) -> str:
	return name.split(".", 1)[0]


def breakdown_parameters_buffers(module: nn.Module) -> dict[str, dict[str, int]]:
	"""Bytes per top-level submodule: params, buffers, and param count."""
	by: dict[str, dict[str, int]] = defaultdict(lambda: {"params": 0, "buffers": 0, "n_params": 0})
	for name, p in module.named_parameters():
		pre = _top_level_prefix(name)
		by[pre]["params"] += p.numel() * p.element_size()
		by[pre]["n_params"] += p.numel()
	for name, b in module.named_buffers():
		pre = _top_level_prefix(name)
		by[pre]["buffers"] += b.numel() * b.element_size()
	return dict(by)


def print_module_storage(title: str, module: nn.Module) -> None:
	parts = breakdown_parameters_buffers(module)
	total_p = sum(v["params"] for v in parts.values())
	total_b = sum(v["buffers"] for v in parts.values())
	print(f"\n=== {title} (parameters + registered buffers) ===")
	print(f"  Total param bytes:   {_fmt_bytes(total_p)}")
	print(f"  Total buffer bytes:  {_fmt_bytes(total_b)}")
	print(f"  Total:               {_fmt_bytes(total_p + total_b)}")
	for name in sorted(parts.keys()):
		v = parts[name]
		sub = v["params"] + v["buffers"]
		print(
			f"  {name:20s}  params {_fmt_bytes(v['params']):>12s}  "
			f"buffers {_fmt_bytes(v['buffers']):>12s}  "
			f"sum {_fmt_bytes(sub):>12s}  ({v['n_params']:,} param elements)"
		)


def print_gradient_storage(module: nn.Module) -> None:
	by: dict[str, int] = defaultdict(int)
	total = 0
	for name, p in module.named_parameters():
		if p.grad is None:
			continue
		g = p.grad
		n = g.numel() * g.element_size()
		by[_top_level_prefix(name)] += n
		total += n
	print(f"\n=== Gradients (.grad) by top-level module ===")
	print(f"  Total grad tensor bytes: {_fmt_bytes(total)}")
	for name in sorted(by.keys()):
		print(f"  {name:20s}  {_fmt_bytes(by[name])}")


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Report GPU memory usage and what it represents.")
	p.add_argument("--device", type=str, default="cuda", help="e.g. cuda or cuda:0")
	p.add_argument("--reset-peak", action="store_true", help="Reset peak stats before optional forward/backward")
	p.add_argument("--no-summary", action="store_true", help="Skip torch.cuda.memory_summary block")
	p.add_argument(
		"--world-model",
		action="store_true",
		help="Construct WorldModel on GPU (loads VAE + UNet like training)",
	)
	p.add_argument("--vae-checkpoint", type=str, default=str(Path("world_model") / "checkpoints" / "vae" / "vae.pt"))
	p.add_argument("--num-actions", type=int, default=18)
	p.add_argument("--history-len", type=int, default=8)
	p.add_argument("--chunk-len", type=int, default=3)
	p.add_argument("--gradient-checkpointing", action="store_true")
	p.add_argument("--batch-size", type=int, default=1)
	p.add_argument(
		"--forward",
		action="store_true",
		help="Run one diffusion_forward in bf16 to measure activation peak",
	)
	p.add_argument(
		"--backward",
		action="store_true",
		help="After --forward, run backward() so .grad and optimizer state can be measured",
	)
	p.add_argument(
		"--optimizer",
		action="store_true",
		help="After loading model, build AdamW(trainable params) and report extra reserved memory",
	)
	p.add_argument(
		"--trainable_parts",
		type=str,
		default="full",
		choices=TRAINABLE_PARTS_CHOICES,
		help="Diffuser finetuning mask (same as train_world_model --trainable_parts).",
	)
	p.add_argument("--unet_top_n_blocks", type=int, default=2)
	p.add_argument("--lora_rank", type=int, default=8)
	p.add_argument("--lora_alpha", type=float, default=8.0)
	p.add_argument("--lora_include_motion", action="store_true")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	if args.backward and not args.forward:
		raise SystemExit("--backward requires --forward")
	if args.optimizer and not args.world_model:
		raise SystemExit("--optimizer requires --world-model")
	if not torch.cuda.is_available():
		raise SystemExit("CUDA is not available.")

	device = torch.device(args.device)
	torch.cuda.set_device(_device_index(device))

	print_driver_memory(device)
	print_driver_vs_pytorch_scope()
	print_allocator_legend(device)
	if not args.no_summary:
		print_torch_memory_summary(device)

	if args.reset_peak:
		torch.cuda.reset_peak_memory_stats(_device_index(device))

	wm: WorldModel | None = None
	opt = None
	if args.world_model:
		wm = WorldModel(
			num_actions=args.num_actions,
			cross_attention_dim=768,
			vae_checkpoint=args.vae_checkpoint,
			prediction_type="epsilon",
			history_len=args.history_len,
			chunk_len=args.chunk_len,
			gradient_checkpointing=args.gradient_checkpointing,
			pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5",
			trainable_parts=args.trainable_parts,
			unet_top_n_blocks=args.unet_top_n_blocks,
			lora_rank=args.lora_rank,
			lora_alpha=args.lora_alpha,
			lora_include_motion=args.lora_include_motion,
		).to(device)
		print_module_storage("WorldModel", wm)
		n_tr, n_tot = count_trainable_params(wm.diffuser)
		print(f"\n=== Trainable subset (diffuser) ===")
		print(f"  trainable_parts={args.trainable_parts}")
		print(f"  trainable scalars: {n_tr:,} / {n_tot:,}")
		if getattr(wm.diffuser, "_attn_lora_injected", False):
			print(f"  spatial-attn LoRA layers: {getattr(wm.diffuser, '_attn_lora_layers', '?')}")

	if args.optimizer and wm is not None:
		before = torch.cuda.memory_allocated(_device_index(device))
		opt = torch.optim.AdamW(wm.trainable_parameters(), lr=1e-4, weight_decay=1e-2)
		after = torch.cuda.memory_allocated(_device_index(device))
		print(f"\n=== Optimizer (AdamW state created on device) ===")
		print(f"  Δ allocated after building optimizer: {_fmt_bytes(after - before)}")
		print("  (exact AdamW footprint shows after first step; run with --forward --backward)")

	if args.forward:
		if wm is None:
			raise SystemExit("--forward requires --world-model")
		B, K, N = args.batch_size, args.history_len, args.chunk_len
		C = wm.latent_channels
		# Infer latent spatial size from VAE (same as scaled latents in training)
		with torch.no_grad():
			dummy_px = torch.zeros(B, 1, 3, 208, 160, device=device)
			z1 = wm.encode_frames(dummy_px[:, 0])
		_, _, h, w = z1.shape
		# Match train_world_model: float latents from VAE, noise in UNet dtype, forward under autocast (bf16).
		z_hist = torch.randn(B, K, C, h, w, device=device, dtype=torch.float32)
		z_tgt = torch.randn(B, N, C, h, w, device=device, dtype=torch.float32)
		ha = torch.zeros(B, K, dtype=torch.long, device=device)
		fa = torch.zeros(B, N, dtype=torch.long, device=device)
		ts = torch.randint(0, wm.num_train_timesteps, (B,), device=device, dtype=torch.long)
		noise = torch.randn(B, N, C, h, w, device=device, dtype=wm.diffuser.unet.dtype)

		torch.cuda.reset_peak_memory_stats(_device_index(device))
		wm.train()
		amp_dtype = torch.bfloat16
		with autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
			out, _ = wm.diffusion_forward(z_hist, z_tgt, ha, fa, ts, noise)
			loss = out.float().pow(2).mean()
		if args.backward:
			loss.backward()
		peak = torch.cuda.max_memory_allocated(_device_index(device))
		print(f"\n=== Forward{'+backward' if args.backward else ''} (bf16, one step) ===")
		print(f"  Latent grid: C={C}, h={h}, w={w}, B={B}, K={K}, N={N}")
		print(f"  Peak allocated this process: {_fmt_bytes(peak)}")
		if args.backward:
			print_gradient_storage(wm)
			if opt is not None:
				step_before = torch.cuda.memory_allocated(_device_index(device))
				opt.step()
				step_after = torch.cuda.memory_allocated(_device_index(device))
				print(f"\n  Δ allocated after optimizer.step(): {_fmt_bytes(step_after - step_before)}")

	if args.world_model or args.forward:
		print("\n=== After measurements (allocator snapshot) ===")
		print_allocator_legend(device)
	else:
		print(
			"\nTip: run with --world-model to load VAE+UNet on GPU; add --forward [--backward] [--optimizer]\n"
			"     for activation/gradient/optimizer peak estimates."
		)


if __name__ == "__main__":
	main()
