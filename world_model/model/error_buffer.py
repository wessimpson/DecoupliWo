from __future__ import annotations

from collections import deque
import torch


class ErrorBuffer:
    """CPU-backed ring buffer of history-shaped residual tensors."""

    def __init__(self, capacity: int = 5000, min_ready_size: int = 128) -> None:
        self.capacity = int(capacity)
        self.min_ready_size = int(min_ready_size)
        self.buf: deque[torch.Tensor] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buf)

    def ready(self) -> bool:
        return len(self.buf) >= self.min_ready_size

    def push(self, delta: torch.Tensor) -> None:
        """
        Store history-shaped residuals.

        Args:
            delta: [B, K, C, h, w]
        """
        if delta.ndim != 5:
            raise ValueError(f"Expected delta [B,K,C,h,w], got shape {tuple(delta.shape)}")

        for d in delta.detach().to("cpu").unbind(0):
            self.buf.append(d)

    def sample_like(self, ref: torch.Tensor) -> torch.Tensor:
        """
        Sample residuals matching ref shape.

        Args:
            ref: [B, K, C, h, w]

        Returns:
            [B, K, C, h, w] on ref device/dtype
        """
        if ref.ndim != 5:
            raise ValueError(f"Expected ref [B,K,C,h,w], got shape {tuple(ref.shape)}")

        B = ref.shape[0]

        if not self.ready():
            return torch.zeros_like(ref)

        idx = torch.randint(len(self.buf), (B,))
        out = torch.stack([self.buf[int(i)] for i in idx], dim=0)
        return out.to(device=ref.device, dtype=ref.dtype)