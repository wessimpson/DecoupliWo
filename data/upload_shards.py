"""
Background uploader that streams finalized ASCII shards from a local root to a remote
root (typically a Google Drive FUSE mount on Colab).

Pairs with ``data/collect_gvgai_ascii.py``: the collector writes
``<local-root>/<split>/<game>/<variant>/shard_XXXXX/`` at wire speed on the VM's local
SSD, and this module rsyncs each shard to the matching path under ``--remote-root`` as
soon as the Java writer finishes it, so the training notebook (which pulls from Drive)
sees frames from every game as collection progresses.

Shard completion signal
-----------------------
``tracks.singlePlayer.ascii.RunAsciiCollectionMCTS$ShardAccumulator.writeShard`` emits
the seven per-shard ``.npy`` files in a fixed order, ending with ``player_y.npy``.
A shard directory is treated as finalized once ``player_y.npy`` exists. After a
successful ``rsync``, we write an empty ``.uploaded`` sentinel inside the local shard
dir so subsequent scans skip it.

Resumability
------------
``rsync --archive`` writes to a temp file and renames on success, so an upload that is
interrupted mid-file does not corrupt the remote shard — the next scan re-runs rsync
from scratch on any shard missing the ``.uploaded`` sentinel.

Typical usage::

	python -m data.upload_shards \\
		--local-root  /content/DecoupliWo/data/transitions \\
		--remote-root /content/drive/MyDrive/DecoupliWo/data/transitions \\
		--poll-seconds 10 \\
		--stop-file /tmp/upload_shards.stop
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SHARD_COMPLETION_MARKER = "player_y.npy"
UPLOADED_SENTINEL = ".uploaded"
DEFAULT_POLL_SECONDS = 10
DEFAULT_STABILITY_SECONDS = 5


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--local-root", type=Path, required=True,
		help="local directory under which <split>/<game>/<variant>/shard_* dirs are being written")
	p.add_argument("--remote-root", type=Path, required=True,
		help="destination root (e.g. /content/drive/MyDrive/DecoupliWo/data/transitions)")
	p.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS,
		help=f"seconds between scans (default: {DEFAULT_POLL_SECONDS})")
	p.add_argument("--stability-seconds", type=int, default=DEFAULT_STABILITY_SECONDS,
		help=("extra wait after the completion marker appears before uploading, to defend "
			f"against a racing fs flush (default: {DEFAULT_STABILITY_SECONDS})"))
	p.add_argument("--stop-file", type=Path, default=None,
		help="when this path exists, finish the current scan and exit cleanly (used by the "
			"collection notebook to signal 'collector finished, flush and stop')")
	p.add_argument("--once", action="store_true",
		help="run a single scan then exit (useful for the final flush after collection)")
	p.add_argument("--rsync-bin", type=str, default="rsync", help="path to rsync executable")
	p.add_argument("--verbose", action="store_true", help="print every upload")
	return p.parse_args()


def find_ready_shards(local_root: Path, stability_seconds: int) -> list[Path]:
	"""Return shard directories that have finished writing and have not yet been uploaded."""
	now = time.time()
	ready: list[Path] = []
	for marker in local_root.rglob(f"shard_*/{SHARD_COMPLETION_MARKER}"):
		shard_dir = marker.parent
		if (shard_dir / UPLOADED_SENTINEL).exists():
			continue
		if now - marker.stat().st_mtime < stability_seconds:
			continue
		ready.append(shard_dir)
	ready.sort()
	return ready


def remote_path_for(local_root: Path, remote_root: Path, shard_dir: Path) -> Path:
	rel = shard_dir.relative_to(local_root)
	return remote_root / rel


def upload_shard(shard_dir: Path, remote_dir: Path, rsync_bin: str, verbose: bool) -> bool:
	"""Rsync ``shard_dir/`` to ``remote_dir/`` and drop a local sentinel on success."""
	remote_dir.parent.mkdir(parents=True, exist_ok=True)
	src = str(shard_dir).rstrip("/") + "/"
	dst = str(remote_dir).rstrip("/") + "/"
	cmd = [rsync_bin, "-a", "--exclude", UPLOADED_SENTINEL, src, dst]
	if verbose:
		print(f"[upload] {shard_dir} -> {remote_dir}")
	rc = subprocess.run(cmd).returncode
	if rc != 0:
		print(f"[upload] rsync failed (rc={rc}) for {shard_dir}", file=sys.stderr)
		return False
	(shard_dir / UPLOADED_SENTINEL).write_text("")
	return True


def scan_and_upload(
	local_root: Path,
	remote_root: Path,
	stability_seconds: int,
	rsync_bin: str,
	verbose: bool,
) -> tuple[int, int]:
	"""Run one upload pass. Returns ``(uploaded, failed)`` counts."""
	ready = find_ready_shards(local_root, stability_seconds)
	uploaded = 0
	failed = 0
	for shard_dir in ready:
		remote_dir = remote_path_for(local_root, remote_root, shard_dir)
		if upload_shard(shard_dir, remote_dir, rsync_bin, verbose):
			uploaded += 1
		else:
			failed += 1
	return uploaded, failed


def should_stop(stop_file: Path | None) -> bool:
	return stop_file is not None and stop_file.exists()


def main() -> int:
	args = parse_args()
	if not args.local_root.is_dir():
		args.local_root.mkdir(parents=True, exist_ok=True)
	args.remote_root.mkdir(parents=True, exist_ok=True)

	print(
		f"[upload] watching {args.local_root} -> {args.remote_root}  "
		f"poll={args.poll_seconds}s  stability={args.stability_seconds}s"
	)

	total_uploaded = 0
	while True:
		uploaded, failed = scan_and_upload(
			args.local_root, args.remote_root, args.stability_seconds, args.rsync_bin, args.verbose,
		)
		total_uploaded += uploaded
		if uploaded or failed:
			print(f"[upload] pass: +{uploaded} uploaded, {failed} failed, total={total_uploaded}")
		if args.once:
			break
		if should_stop(args.stop_file):
			uploaded, failed = scan_and_upload(
				args.local_root, args.remote_root, 0, args.rsync_bin, args.verbose,
			)
			total_uploaded += uploaded
			print(f"[upload] stop signal seen; final flush uploaded {uploaded} more (total={total_uploaded})")
			break
		time.sleep(args.poll_seconds)

	print(f"[upload] done. total uploaded = {total_uploaded}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
