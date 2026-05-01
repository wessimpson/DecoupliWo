"""
Smoke test: WAN VAE compatibility with existing (stale) shards.

Simulates the teammate's scenario:
1. Creates tiny synthetic raw shards (obs.npy + action.npy + n_actions.npy)
2. Creates STALE encoded shards with C=4 (old SD VAE shape)
3. Runs encode_transition.py which should auto-detect the stale C=4 -> re-encode to C=16
4. Verifies the new latent.npy has correct shape
5. Runs train_dynamics.py for 2 steps to confirm end-to-end training works

Run:  python -m tests.smoke_wan_vae_compat
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


PYTHON = sys.executable
ROOT = Path(__file__).resolve().parent.parent
SMOKE_ROOT = ROOT / "data" / "transitions" / "_smoke_test"
N_FRAMES = 12   # tiny shard: 12 frames
H, W = 64, 64   # small, divisible by 8
N_ACTIONS = 7
OLD_LATENT_C = 4   # simulate old SD VAE latent channels
EXPECTED_C = 16    # WAN VAE z_dim


def _create_raw_shards() -> None:
    """Create raw obs.npy + action.npy for train and test splits."""
    for split in ("train", "test"):
        env_dir = SMOKE_ROOT / split / "smoke_env"
        shard = env_dir / "shard_000"
        shard.mkdir(parents=True, exist_ok=True)
        obs = np.random.randint(0, 256, (N_FRAMES, H, W, 3), dtype=np.uint8)
        act = np.random.randint(0, N_ACTIONS, (N_FRAMES,), dtype=np.int64)
        np.save(shard / "obs.npy", obs)
        np.save(shard / "action.npy", act)
        np.save(shard / "n_actions.npy", np.array(N_ACTIONS))
    print("[smoke] raw shards created")


def _create_stale_encoded_shards() -> None:
    """Simulate teammate's OLD encoded latent.npy with C=4 (wrong channels)."""
    for split in ("train", "test"):
        enc_dir = SMOKE_ROOT / "encoded" / split / "smoke_env" / "shard_000"
        enc_dir.mkdir(parents=True, exist_ok=True)
        # Stale latent: C=4 instead of expected C=16
        stale_lat = np.random.randn(N_FRAMES, OLD_LATENT_C, H // 8, W // 8).astype(np.float16)
        np.save(enc_dir / "latent.npy", stale_lat)
        # Copy action/n_actions too
        src_shard = SMOKE_ROOT / split / "smoke_env" / "shard_000"
        shutil.copy2(src_shard / "action.npy", enc_dir / "action.npy")
        shutil.copy2(src_shard / "n_actions.npy", enc_dir / "n_actions.npy")
    print(f"[smoke] stale encoded shards created (C={OLD_LATENT_C})")


def _verify_stale_latent() -> None:
    """Verify stale latent has wrong channels before re-encoding."""
    lat = np.load(SMOKE_ROOT / "encoded" / "train" / "smoke_env" / "shard_000" / "latent.npy")
    assert lat.shape[1] == OLD_LATENT_C, f"Expected stale C={OLD_LATENT_C}, got {lat.shape[1]}"
    print(f"[smoke] confirmed stale latent shape: {lat.shape}  (C={OLD_LATENT_C})")


def _run_encode() -> None:
    """Run encode_transition.py -- should detect stale shards and re-encode."""
    cmd = [
        PYTHON, "-m", "world_model.encode_transition",
        "--transitions_root", str(SMOKE_ROOT),
        "--split", "both",
        "--batch_size", "4",
    ]
    print(f"[smoke] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(f"encode_transition.py failed with exit code {result.returncode}")
    print("[smoke] encode_transition.py succeeded")


def _verify_encoded_latent() -> None:
    """Verify re-encoded latent has correct WAN VAE channels."""
    for split in ("train", "test"):
        lat_path = SMOKE_ROOT / "encoded" / split / "smoke_env" / "shard_000" / "latent.npy"
        assert lat_path.exists(), f"Missing {lat_path}"
        lat = np.load(lat_path)
        print(f"[smoke] {split} encoded latent shape: {lat.shape}")
        assert lat.shape[0] == N_FRAMES, f"Expected N={N_FRAMES}, got {lat.shape[0]}"
        assert lat.shape[1] == EXPECTED_C, (
            f"FAIL: Expected C={EXPECTED_C} (WAN VAE), got C={lat.shape[1]}. "
            f"Stale shard was not re-encoded!"
        )
    print(f"[smoke] PASS: all encoded latents have correct shape C={EXPECTED_C}")


def _run_train_dynamics(max_steps: int = 2) -> None:
    """Run train_dynamics.py for a few steps to confirm end-to-end works."""
    ckpt_dir = SMOKE_ROOT / "_ckpt"
    log_dir = SMOKE_ROOT / "_logs"
    cmd = [
        PYTHON, "-m", "world_model.train_dynamics",
        "--transitions_root", str(SMOKE_ROOT),
        "--encoded_subdir", "encoded",
        "--env", "smoke_env",
        "--num_actions", str(N_ACTIONS),
        "--context_len", "2",
        "--batch_size", "2",
        "--max_train_steps", str(max_steps),
        "--num_train_epochs", "1",
        "--validation_every", "0",
        "--save_every", "0",
        "--checkpoint_dir", str(ckpt_dir),
        "--log_dir", str(log_dir),
        "--mixed_precision", "no",
        "--num_workers", "0",
        "--lr_warmup_steps", "0",
    ]
    print(f"[smoke] running train_dynamics for {max_steps} steps ...")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-3000:])
        raise RuntimeError(f"train_dynamics.py failed with exit code {result.returncode}")
    print(f"[smoke] PASS: train_dynamics ran {max_steps} steps successfully")


def _cleanup() -> None:
    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)
    print("[smoke] cleaned up")


def main() -> None:
    print("=" * 60)
    print("SMOKE TEST: WAN VAE compatibility with stale encoded shards")
    print("=" * 60)

    try:
        # Clean slate
        _cleanup()

        # Step 1: Create raw data
        _create_raw_shards()

        # Step 2: Create stale encoded shards (simulating old SD VAE C=4)
        _create_stale_encoded_shards()
        _verify_stale_latent()

        # Step 3: Run encoder -- should detect stale and re-encode
        _run_encode()

        # Step 4: Verify re-encoded latents have correct shape
        _verify_encoded_latent()

        # Step 5: Run training for 2 steps
        _run_train_dynamics(max_steps=2)

        print("\n" + "=" * 60)
        print("ALL SMOKE TESTS PASSED")
        print("=" * 60)

    finally:
        _cleanup()


if __name__ == "__main__":
    main()
