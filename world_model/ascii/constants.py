"""Shared constants for the ASCII frame representation.

Frames are stored as ``uint8[N, H, W]`` arrays of printable ASCII bytes
(identical in spirit to a VGDL level file, one character per tile).
The byte value doubles as the token id for the categorical VAE, so the
vocabulary size is fixed at 256 and no lookup table is needed at train time.
``PAD_BYTE`` is reserved for padding a game's native grid up to the unified
``CANVAS_H`` × ``CANVAS_W`` canvas; it is never emitted by a game.
"""

from __future__ import annotations

VOCAB_SIZE: int = 256

PAD_BYTE: int = 0

CANVAS_H: int = 16
CANVAS_W: int = 32
