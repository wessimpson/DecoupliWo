"""Load (frames, actions) rollouts for world-model training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class RolloutClipDataset(Dataset):
    """
    Expects ``*.npz`` with:
      - ``frames``: uint8 [T, H, W, 3]
      - ``actions``: int64 [T] (ALE discrete ids)
      - ``n_actions``: int scalar array (optional; else inferred as actions.max()+1)
    Yields a clip of length ``seq_len`` with frames resized to ``image_size``.
    """

    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int,
        image_size: int = 84,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.image_size = image_size
        self.stride = max(1, stride)
        self._files = sorted(self.data_dir.glob("*.npz"))
        if not self._files:
            raise FileNotFoundError(f"No .npz under {self.data_dir}")
        self._index: list[tuple[Path, int, int]] = []
        self.n_actions = 0
        for fp in self._files:
            with np.load(fp, mmap_mode="r") as z:
                t = int(z["frames"].shape[0])
                na = int(z["n_actions"][0]) if "n_actions" in z else int(np.max(z["actions"])) + 1
                self.n_actions = max(self.n_actions, na)
                for s in range(0, t - seq_len + 1, self.stride):
                    self._index.append((fp, s, na))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path, start, _na = self._index[idx]
        z = np.load(path)
        frames = z["frames"][start : start + self.seq_len].astype(np.float32) / 127.5 - 1.0
        actions = z["actions"][start : start + self.seq_len].astype(np.int64)
        z.close()
        x = torch.from_numpy(frames).permute(0, 3, 1, 2)
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return {
            "frames": x,
            "actions": torch.from_numpy(actions).long(),
        }
