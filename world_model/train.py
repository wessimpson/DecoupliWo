"""
Train causal flow model on (frame, action) clips.

VAE options:
  - ``atari``: train small ``AtariFrameVAE`` from data, then train flow on its latents.
  - ``wan``: frozen pretrained Wan 2.1 VAE (``Wan2.1_VAE.pth`` in ``world_model/checkpoint/vae/``).

Checkpoints:
  - ``world_model/checkpoint/vae/`` — place ``Wan2.1_VAE.pth`` here; Atari VAE saves as ``atari_vae.pt``.
  - ``world_model/checkpoint/dit/`` — flow weights ``flow.pt`` and ``config.pt``.

Run from repo root:
  python -m world_model.collect_data --episodes 300
  python -m world_model.train --data_dir data/space_invaders_rollouts --vae atari
  python -m world_model.train --data_dir data/space_invaders_rollouts --vae wan

Requires PyTorch. Wan path needs ``Matrix-Game-2`` in the repo and that package's deps (einops, etc.).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from world_model.atari_vae import AtariFrameVAE
from world_model.causal_latent_dit import CausalLatentFlowModel
from world_model.dataset import RolloutClipDataset
from world_model.wan_vae_encoder import FrozenWanLatentEncoder

_WM_DIR = Path(__file__).resolve().parent
_CHECKPOINT_ROOT = _WM_DIR / "checkpoint"
_VAE_DIR = _CHECKPOINT_ROOT / "vae"
_DIT_DIR = _CHECKPOINT_ROOT / "dit"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/space_invaders_rollouts")
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=84, help="Dataset resize (Wan encoder re-resizes internally).")
    ap.add_argument("--vae", type=str, choices=("atari", "wan"), default="atari")
    ap.add_argument(
        "--wan_vae_dir",
        type=str,
        default=str(_VAE_DIR),
        help="Folder with Wan2.1_VAE.pth (default: world_model/checkpoint/vae).",
    )
    ap.add_argument("--wan_image_size", type=int, default=256, help="Must be divisible by 8.")
    ap.add_argument("--wan_pool_size", type=int, default=4, help="Adaptive pool on latent grid; latent_dim = 16 * pool^2.")
    ap.add_argument("--latent_dim", type=int, default=128, help="Only used for --vae atari.")
    ap.add_argument("--vae_steps", type=int, default=3000)
    ap.add_argument("--flow_steps", type=int, default=8000)
    ap.add_argument("--lr_vae", type=float, default=3e-4)
    ap.add_argument("--lr_flow", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--save_dir",
        type=str,
        default=str(_DIT_DIR),
        help="Directory for flow.pt and config.pt (default: world_model/checkpoint/dit).",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    vae_dir = Path(args.wan_vae_dir).resolve()
    save_dir = Path(args.save_dir).resolve()
    _VAE_DIR.mkdir(parents=True, exist_ok=True)
    _DIT_DIR.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    ds = RolloutClipDataset(args.data_dir, seq_len=args.seq_len, image_size=args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    n_actions = ds.n_actions

    vae_atari: AtariFrameVAE | None = None
    wan_enc: FrozenWanLatentEncoder | None = None
    latent_dim: int

    if args.vae == "atari":
        latent_dim = args.latent_dim
        vae_atari = AtariFrameVAE(image_size=args.image_size, latent_dim=latent_dim).to(device)
    else:
        wan_enc = FrozenWanLatentEncoder(
            pretrained_dir=vae_dir,
            image_size=args.wan_image_size,
            pool_size=args.wan_pool_size,
        ).to(device)
        if device.type == "cuda":
            wan_enc.wan.to(dtype=torch.bfloat16)
        latent_dim = wan_enc.latent_dim

    flow = CausalLatentFlowModel(latent_dim=latent_dim, n_actions=n_actions).to(device)

    opt_vae = (
        torch.optim.AdamW(vae_atari.parameters(), lr=args.lr_vae) if vae_atari is not None else None
    )
    opt_flow = torch.optim.AdamW(flow.parameters(), lr=args.lr_flow)

    # --- VAE phase (Atari only); weights under checkpoint/vae/ ---
    if vae_atari is not None and args.vae_steps > 0:
        step = 0
        vae_atari.train()
        while step < args.vae_steps:
            for batch in dl:
                if step >= args.vae_steps:
                    break
                x = batch["frames"].to(device)
                b, t, c, h, w = x.shape
                x_flat = x.reshape(b * t, c, h, w)
                opt_vae.zero_grad(set_to_none=True)
                loss = vae_atari.loss_recon_kl(x_flat)
                loss.backward()
                opt_vae.step()
                step += 1
                if step % 200 == 0:
                    print(f"vae step {step} loss {loss.item():.4f}")
        atari_path = _VAE_DIR / "atari_vae.pt"
        torch.save(vae_atari.state_dict(), atari_path)
        vae_atari.eval()
        print(f"saved Atari VAE to {atari_path}")

    # --- Flow (DiT) phase; weights under checkpoint/dit/ by default ---
    step = 0
    flow.train()
    while step < args.flow_steps:
        for batch in dl:
            if step >= args.flow_steps:
                break
            x = batch["frames"].to(device)
            act = batch["actions"].to(device).clamp(0, n_actions - 1)
            b, t, c, h, w = x.shape

            with torch.no_grad():
                if vae_atari is not None:
                    x_flat = x.reshape(b * t, c, h, w)
                    mu, _logvar = vae_atari.encode(x_flat)
                    z0 = mu.view(b, t, -1)
                else:
                    assert wan_enc is not None
                    z0 = wan_enc.encode_frames(x, device=device).float()

            eps = torch.randn_like(z0)
            tdiff = torch.rand(b, device=device)
            t_b = tdiff.view(b, 1, 1)
            z_noisy = (1.0 - t_b) * z0 + t_b * eps
            target_v = eps - z0

            opt_flow.zero_grad(set_to_none=True)
            pred = flow(z_noisy, tdiff, act)
            loss = F.mse_loss(pred, target_v)
            loss.backward()
            opt_flow.step()
            step += 1
            if step % 200 == 0:
                print(f"flow step {step} loss {loss.item():.4f}")

    torch.save(flow.state_dict(), save_dir / "flow.pt")
    cfg = {
        "vae": args.vae,
        "n_actions": n_actions,
        "seq_len": args.seq_len,
        "image_size": args.image_size,
        "latent_dim": latent_dim,
        "vae_checkpoint_dir": str(_VAE_DIR.resolve()),
        "dit_checkpoint_dir": str(save_dir.resolve()),
    }
    if args.vae == "wan":
        cfg["wan_vae_dir"] = str(vae_dir)
        cfg["wan_image_size"] = args.wan_image_size
        cfg["wan_pool_size"] = args.wan_pool_size
    else:
        cfg["atari_vae_file"] = str((_VAE_DIR / "atari_vae.pt").resolve())
    torch.save(cfg, save_dir / "config.pt")
    print(f"saved flow + config to {save_dir}")


if __name__ == "__main__":
    main()
