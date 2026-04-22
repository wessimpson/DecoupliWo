"""Runtime utilities for ASCII frames.

Three jobs:
	1. Pad a game's native grid up to the unified canvas (``pad_to_canvas``).
	2. Crop the canvas back to a game's native grid after decoding (``crop_from_canvas``).
	3. Human-readable dump of a frame as text (``dump_ascii``).

All functions work on ``uint8[H, W]`` or ``uint8[N, H, W]`` numpy arrays; no
tensor conversions are done here. The byte value is also the VAE's token id.
"""

from __future__ import annotations

from typing import overload

import numpy as np

from world_model.ascii.constants import CANVAS_H, CANVAS_W, PAD_BYTE


def _assert_grid(arr: np.ndarray) -> None:
	assert arr.dtype == np.uint8, f"expected uint8 grid, got {arr.dtype}"
	assert arr.ndim in (2, 3), f"expected [H,W] or [N,H,W], got shape {arr.shape}"


def pad_to_canvas(
	grid: np.ndarray,
	canvas_h: int = CANVAS_H,
	canvas_w: int = CANVAS_W,
	pad_byte: int = PAD_BYTE,
) -> np.ndarray:
	"""Pad a ``[H, W]`` or ``[N, H, W]`` uint8 grid up to ``[canvas_h, canvas_w]`` with ``pad_byte``."""
	_assert_grid(grid)
	if grid.ndim == 2:
		h, w = grid.shape
		assert h <= canvas_h and w <= canvas_w, f"grid {h}x{w} exceeds canvas {canvas_h}x{canvas_w}"
		out = np.full((canvas_h, canvas_w), pad_byte, dtype=np.uint8)
		out[:h, :w] = grid
		return out

	n, h, w = grid.shape
	assert h <= canvas_h and w <= canvas_w, f"grid {h}x{w} exceeds canvas {canvas_h}x{canvas_w}"
	out = np.full((n, canvas_h, canvas_w), pad_byte, dtype=np.uint8)
	out[:, :h, :w] = grid
	return out


def crop_from_canvas(canvas: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
	"""Inverse of :func:`pad_to_canvas` when the native ``orig_h``/``orig_w`` are known."""
	_assert_grid(canvas)
	if canvas.ndim == 2:
		return canvas[:orig_h, :orig_w].copy()
	return canvas[:, :orig_h, :orig_w].copy()


@overload
def dump_ascii(grid: np.ndarray) -> str: ...
@overload
def dump_ascii(grid: np.ndarray, pad_char: str) -> str: ...


def dump_ascii(grid: np.ndarray, pad_char: str = " ") -> str:
	"""Render a single ``[H, W]`` frame as a newline-joined string.

	Non-printable bytes (including ``PAD_BYTE``) are replaced with ``pad_char``
	so a shard can be eyeballed with a plain ``print``.
	"""
	assert grid.dtype == np.uint8 and grid.ndim == 2, f"dump_ascii expects [H,W] uint8, got {grid.shape}/{grid.dtype}"
	assert len(pad_char) == 1, "pad_char must be a single character"

	pad_ord = ord(pad_char)
	safe = np.where((grid >= 0x20) & (grid < 0x7F), grid, np.uint8(pad_ord))
	return "\n".join(row.tobytes().decode("ascii") for row in safe)
