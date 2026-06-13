"""TensorBoard / matplotlib image helpers."""

from __future__ import annotations

import torch


def tensor_to_imshow01(t: torch.Tensor) -> "torch.Tensor":
	"""``[3,H,W]`` in ``[-1,1]`` → ``[H,W,3]`` float32 ``[0,1]`` (numpy via .numpy() in callers)."""
	return ((t.clamp(-1, 1) + 1) * 0.5).cpu().permute(1, 2, 0).numpy().astype("float32")


def neg1_to_01(chw: torch.Tensor) -> torch.Tensor:
	"""``[1,3,H,W]`` in ``[-1,1]`` → ``[0,1]``."""
	return ((chw.clamp(-1, 1) + 1) * 0.5).float()


def strip_horizontal(
	chunks: list[torch.Tensor],
	*,
	gap_px: int = 6,
	gap_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
	"""Concatenate ``[1,3,H,W]`` in ``[0,1]`` along width."""
	if not chunks:
		raise ValueError("need at least one chunk")
	if len(chunks) == 1:
		return chunks[0]
	ref = chunks[0]
	H, device, dtype = int(ref.shape[-2]), ref.device, ref.dtype
	g = max(0, int(gap_px))
	if g == 0:
		return torch.cat(chunks, dim=-1)
	r0, g0, b0 = gap_rgb
	sep = torch.tensor([[[r0]], [[g0]], [[b0]]], device=device, dtype=dtype).expand(1, 3, H, g)
	parts = [chunks[0]]
	for ch in chunks[1:]:
		parts.extend([sep, ch])
	return torch.cat(parts, dim=-1)


def vstack_tgt_gen(tgt_01: torch.Tensor, gen_01: torch.Tensor, *, gap_px: int = 4) -> torch.Tensor:
	"""Target above generated; ``[1,3,H,W]`` in ``[0,1]``."""
	_, _, _, W = tgt_01.shape
	g = max(0, int(gap_px))
	if g == 0:
		return torch.cat([tgt_01, gen_01], dim=-2)
	sep = torch.ones(1, 3, g, W, dtype=tgt_01.dtype, device=tgt_01.device)
	return torch.cat([tgt_01, sep, gen_01], dim=-2)


def batched_ranges(n: int, chunk: int) -> list[tuple[int, int]]:
	c = max(1, int(chunk))
	return [(s, min(s + c, n)) for s in range(0, n, c)]
