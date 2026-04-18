"""Render an ASCII frame as the real GVGAI game via a persistent Java renderer.

The Java side is :class:`tracks.singlePlayer.rendering.AsciiRenderServer`
(``gvgai_java_stubs/src/tracks/singlePlayer/rendering/AsciiRenderServer.java``
in this repo; the colleague owning the ``gvgai/`` submodule should copy that
file into the submodule's matching path). This module launches the server as
a subprocess, speaks its line protocol over TCP, and returns an RGB ``uint8``
numpy array per call.

Typical usage::

	with GvgaiRenderer(gvgai_root=Path("gvgai"), game="aliens") as r:
		rgb = r.render(ascii_grid)  # ascii_grid: np.ndarray[H,W] uint8
"""

from __future__ import annotations

import io
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from world_model.ascii.constants import PAD_BYTE
from world_model.ascii.tokenizer import crop_from_canvas

SERVER_MAIN_CLASS: str = "tracks.singlePlayer.rendering.AsciiRenderServer"
SERVER_READY_PREFIX: str = "AsciiRenderServer listening on "
SERVER_STARTUP_TIMEOUT_S: float = 10.0
SERVER_RECV_CHUNK: int = 64 * 1024


def _default_classpath(gvgai_root: Path) -> str:
	"""Best-effort classpath for the GVGAI build tree.

	Points at the conventional IntelliJ/ant output directory (``out/production/gvgai``).
	Callers can override via the ``GVGAI_CLASSPATH`` environment variable.
	"""
	env = os.environ.get("GVGAI_CLASSPATH")
	if env:
		return env
	return str(gvgai_root / "out" / "production" / "gvgai")


def _default_vgdl_path(gvgai_root: Path, game: str) -> Path:
	return gvgai_root / "examples" / "gridphysics" / f"{game}.txt"


class GvgaiRendererError(RuntimeError):
	pass


class GvgaiRenderer:
	"""Spawns an AsciiRenderServer JVM and renders ASCII grids to RGB.

	One renderer instance is bound to a single game (VGDL file) to keep the
	protocol simple; construct another instance for a different game.
	"""

	def __init__(
		self,
		gvgai_root: Path,
		game: str,
		*,
		classpath: str | None = None,
		vgdl_path: Path | None = None,
		host: str = "127.0.0.1",
		port: int = 0,
		java_bin: str = "java",
		extra_jvm_args: tuple[str, ...] = (),
	) -> None:
		self.gvgai_root = Path(gvgai_root)
		self.game = str(game)
		self.classpath = classpath or _default_classpath(self.gvgai_root)
		self.vgdl_path = Path(vgdl_path or _default_vgdl_path(self.gvgai_root, self.game))
		self.host = host
		self.requested_port = int(port)
		self.java_bin = str(java_bin)
		self.extra_jvm_args = tuple(extra_jvm_args)

		self._proc: subprocess.Popen | None = None
		self._sock: socket.socket | None = None
		self._reader: io.BufferedReader | None = None
		self._bound_port: int = -1
		self.screen_w: int = 0
		self.screen_h: int = 0
		self.block_size: int = 0

	def __enter__(self) -> GvgaiRenderer:
		self.start()
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		self.close()

	def start(self) -> None:
		if self._proc is not None:
			return
		if not self.vgdl_path.is_file():
			raise GvgaiRendererError(f"VGDL file not found: {self.vgdl_path}")

		cmd = [self.java_bin, *self.extra_jvm_args, "-cp", self.classpath, SERVER_MAIN_CLASS, str(self.requested_port)]
		self._proc = subprocess.Popen(
			cmd,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			cwd=str(self.gvgai_root),
			bufsize=1,
			text=True,
		)
		self._bound_port = self._wait_for_ready_port()
		self._connect()
		self._init_game()

	def close(self) -> None:
		try:
			if self._sock is not None:
				try:
					self._send_line("QUIT")
					self._read_line()
				except Exception:
					pass
				self._sock.close()
		finally:
			self._sock = None
			self._reader = None
			proc = self._proc
			self._proc = None
			if proc is not None and proc.poll() is None:
				proc.terminate()
				try:
					proc.wait(timeout=2.0)
				except subprocess.TimeoutExpired:
					proc.kill()

	def render(self, ascii_grid: np.ndarray) -> np.ndarray:
		"""Render ``ascii_grid`` (``[H,W] uint8``) to RGB ``[H*block, W*block, 3] uint8``.

		The grid may contain :data:`PAD_BYTE` outside the game's native region;
		pad cells are cropped before sending to the Java side to avoid confusing
		VGDL's level parser (which expects only chars declared in ``LevelMapping``).
		"""
		assert self._sock is not None, "renderer not started; call start() or use as a context manager"
		assert ascii_grid.dtype == np.uint8 and ascii_grid.ndim == 2, (
			f"expected uint8[H,W], got dtype={ascii_grid.dtype} ndim={ascii_grid.ndim}"
		)
		grid = _crop_pad_region(ascii_grid)
		rows = grid.shape[0]
		payload_header = f"RENDER 0 {rows}"
		lines = [payload_header]
		for r in range(rows):
			lines.append(grid[r].tobytes().decode("ascii"))
		self._send_lines(lines)

		resp = self._read_line()
		tokens = resp.split()
		if not tokens or tokens[0] != "OK":
			raise GvgaiRendererError(f"render failed: {resp!r}")
		n_bytes = int(tokens[1])
		png_bytes = self._read_exact(n_bytes)
		return _png_bytes_to_rgb_array(png_bytes)

	def _wait_for_ready_port(self) -> int:
		assert self._proc is not None
		deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT_S
		assert self._proc.stdout is not None
		while time.monotonic() < deadline:
			line = self._proc.stdout.readline()
			if not line:
				if self._proc.poll() is not None:
					err = self._proc.stderr.read() if self._proc.stderr is not None else ""
					raise GvgaiRendererError(
						f"AsciiRenderServer exited before ready (code={self._proc.returncode}). "
						f"stderr:\n{err}"
					)
				time.sleep(0.02)
				continue
			line = line.strip()
			if line.startswith(SERVER_READY_PREFIX):
				return int(line[len(SERVER_READY_PREFIX):].strip())
		raise GvgaiRendererError(
			f"Timed out after {SERVER_STARTUP_TIMEOUT_S}s waiting for AsciiRenderServer to bind a port."
		)

	def _connect(self) -> None:
		sock = socket.create_connection((self.host, self._bound_port))
		sock.settimeout(30.0)
		self._sock = sock
		self._reader = sock.makefile("rb")

	def _init_game(self) -> None:
		self._send_line(f"INIT {self.vgdl_path.resolve()}")
		resp = self._read_line()
		tokens = resp.split()
		if not tokens or tokens[0] != "OK":
			raise GvgaiRendererError(f"INIT failed: {resp!r}")
		self.screen_w = int(tokens[1])
		self.screen_h = int(tokens[2])
		self.block_size = int(tokens[3])

	def _send_line(self, s: str) -> None:
		self._send_lines([s])

	def _send_lines(self, lines: list[str]) -> None:
		assert self._sock is not None
		blob = ("\n".join(lines) + "\n").encode("ascii")
		self._sock.sendall(blob)

	def _read_line(self) -> str:
		assert self._reader is not None
		raw = self._reader.readline()
		if not raw:
			raise GvgaiRendererError("connection closed by AsciiRenderServer")
		return raw.decode("ascii", errors="replace").rstrip("\r\n")

	def _read_exact(self, n: int) -> bytes:
		assert self._reader is not None
		out = bytearray()
		while len(out) < n:
			chunk = self._reader.read(min(SERVER_RECV_CHUNK, n - len(out)))
			if not chunk:
				raise GvgaiRendererError("connection closed mid-payload")
			out.extend(chunk)
		return bytes(out)


def _crop_pad_region(grid: np.ndarray) -> np.ndarray:
	"""Drop trailing PAD rows/columns so the Java parser only sees real chars."""
	non_pad = grid != PAD_BYTE
	if not non_pad.any():
		return grid
	row_mask = non_pad.any(axis=1)
	col_mask = non_pad.any(axis=0)
	last_row = int(np.where(row_mask)[0][-1]) + 1
	last_col = int(np.where(col_mask)[0][-1]) + 1
	return np.ascontiguousarray(grid[:last_row, :last_col])


def _png_bytes_to_rgb_array(png_bytes: bytes) -> np.ndarray:
	"""PNG -> ``uint8[H,W,3]`` using the lightest available PIL dep (pillow)."""
	from PIL import Image

	with Image.open(io.BytesIO(png_bytes)) as img:
		rgb = img.convert("RGB")
		return np.asarray(rgb, dtype=np.uint8).copy()


@contextmanager
def open_renderer(
	gvgai_root: Path,
	game: str,
	**kwargs,
) -> Iterator[GvgaiRenderer]:
	"""Convenience wrapper so callers can ``with open_renderer(...) as r:``."""
	renderer = GvgaiRenderer(gvgai_root=gvgai_root, game=game, **kwargs)
	try:
		renderer.start()
		yield renderer
	finally:
		renderer.close()


__all__ = [
	"GvgaiRenderer",
	"GvgaiRendererError",
	"open_renderer",
	"crop_from_canvas",
]
