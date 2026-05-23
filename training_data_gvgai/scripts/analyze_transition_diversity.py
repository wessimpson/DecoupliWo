#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def infer_rule_tag(env: str) -> str:
    if "_rules_" not in env:
        return "base"
    return env.split("_rules_", 1)[1]


def infer_level_index(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value)
    match = re.search(r"lvl(\d+)", text)
    return match.group(1) if match else text


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def iter_split_dirs(root: Path) -> Iterable[tuple[str, Path]]:
    named = [name for name in ("train", "test", "val") if (root / name).is_dir()]
    if named:
        for name in named:
            yield name, root / name
        return
    yield root.name, root


def iter_shards(root: Path) -> Iterable[tuple[str, str, Path]]:
    for split, split_dir in iter_split_dirs(root):
        for env_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for shard_dir in sorted(env_dir.glob("shard_*")):
                if (shard_dir / "obs.npy").is_file() and (shard_dir / "action.npy").is_file():
                    yield split, env_dir.name, shard_dir


def entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log2(p)).sum())


def ensure_action_counts(counts: np.ndarray, max_action: int) -> np.ndarray:
    if max_action < len(counts):
        return counts
    out = np.zeros(max_action + 1, dtype=np.int64)
    out[: len(counts)] = counts
    return out


@dataclass
class DiversityStats:
    dataset: str
    split: str
    env: str
    profile: str = "all"
    level: str = "all"
    frames: int = 0
    shards: int = 0
    obs_shape: str = ""
    obs_dtype: str = ""
    profiles: set[str] = field(default_factory=set)
    levels: set[str] = field(default_factory=set)
    rule_tags: set[str] = field(default_factory=set)
    n_actions_values: set[int] = field(default_factory=set)
    action_counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    restarted_true: int = 0
    restarted_total: int = 0
    sampled_frames: int = 0
    nonblank_frames: int = 0
    frame_std_sum: float = 0.0
    frame_delta_sum: float = 0.0
    frame_delta_count: int = 0
    frame_hashes: set[bytes] = field(default_factory=set)
    player_samples: int = 0
    player_grid_bins: int = 20
    player_grid_cells: set[tuple[int, int]] = field(default_factory=set)
    player_positions: set[tuple[int, int]] = field(default_factory=set)
    player_position_cap_reached: bool = False

    def add_shard_meta(self, obs: np.ndarray, metadata: dict[str, Any], env: str) -> None:
        self.shards += 1
        self.frames += int(obs.shape[0])
        if not self.obs_shape:
            self.obs_shape = "x".join(str(x) for x in obs.shape[1:])
            self.obs_dtype = str(obs.dtype)
        profile = str(metadata.get("profile") or "unknown")
        level = str(metadata.get("level_index") or infer_level_index(metadata.get("level_file")))
        rule_tag = str(metadata.get("source_rule_tag") or infer_rule_tag(env))
        self.profiles.add(profile)
        self.levels.add(level)
        self.rule_tags.add(rule_tag)

    def add_actions(self, actions: np.ndarray, n_actions: int | None) -> None:
        values = np.asarray(actions, dtype=np.int64).reshape(-1)
        if values.size:
            max_action = int(values.max())
            self.action_counts = ensure_action_counts(self.action_counts, max_action)
            self.action_counts += np.bincount(values, minlength=len(self.action_counts))
        if n_actions is not None:
            self.n_actions_values.add(int(n_actions))

    def add_restarted(self, values: np.ndarray | None) -> None:
        if values is None:
            return
        arr = np.asarray(values).reshape(-1)
        self.restarted_true += int(arr.astype(bool).sum())
        self.restarted_total += int(arr.size)

    def add_player_positions(self, xs: np.ndarray | None, ys: np.ndarray | None, cap: int) -> None:
        if xs is None or ys is None:
            return
        x = np.asarray(xs, dtype=np.float64).reshape(-1)
        y = np.asarray(ys, dtype=np.float64).reshape(-1)
        if x.size == 0 or y.size == 0:
            return
        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            return
        self.player_samples += int(x.size)
        if not self.player_position_cap_reached:
            for px, py in zip(x.tolist(), y.tolist()):
                self.player_positions.add((int(round(px)), int(round(py))))
                if len(self.player_positions) >= cap:
                    self.player_position_cap_reached = True
                    break

    def add_sampled_frames(self, frames: np.ndarray, hash_stride: int, sample_cap: int) -> None:
        if self.sampled_frames >= sample_cap:
            return
        remaining = sample_cap - self.sampled_frames
        frames = np.asarray(frames)
        if frames.shape[0] > remaining:
            frames = frames[:remaining]
        prev: np.ndarray | None = None
        stride = max(1, int(hash_stride))
        for frame in frames:
            rgb = frame[..., :3].astype(np.uint8, copy=False)
            self.sampled_frames += 1
            std = float(rgb.std())
            self.frame_std_sum += std
            if std > 1.0:
                self.nonblank_frames += 1
            small = np.ascontiguousarray(rgb[::stride, ::stride])
            self.frame_hashes.add(hashlib.blake2b(small.tobytes(), digest_size=8).digest())
            if prev is not None and prev.shape == rgb.shape:
                self.frame_delta_sum += float(np.mean(np.abs(rgb.astype(np.int16) - prev.astype(np.int16))))
                self.frame_delta_count += 1
            prev = rgb

    def row(self) -> dict[str, Any]:
        n_actions = max(self.n_actions_values) if self.n_actions_values else len(self.action_counts)
        nonzero_actions = int(np.count_nonzero(self.action_counts))
        action_entropy = entropy_from_counts(self.action_counts)
        player_grid_coverage = self.compute_player_grid_coverage()
        return {
            "dataset": self.dataset,
            "split": self.split,
            "env": self.env,
            "profile": self.profile,
            "level": self.level,
            "frames": self.frames,
            "shards": self.shards,
            "obs_shape": self.obs_shape,
            "obs_dtype": self.obs_dtype,
            "profiles_seen": "|".join(sorted(self.profiles)),
            "levels_seen": "|".join(sorted(self.levels, key=lambda x: (not x.isdigit(), x))),
            "rule_tags_seen": "|".join(sorted(self.rule_tags)),
            "n_actions": "|".join(str(x) for x in sorted(self.n_actions_values)),
            "action_entropy_bits": round(action_entropy, 6),
            "action_coverage": round(nonzero_actions / max(n_actions, 1), 6),
            "nonzero_actions": nonzero_actions,
            "action_counts_json": json.dumps(self.action_counts.tolist(), separators=(",", ":")),
            "restart_rate": round(self.restarted_true / self.restarted_total, 6) if self.restarted_total else "",
            "sampled_frames": self.sampled_frames,
            "nonblank_rate": round(self.nonblank_frames / self.sampled_frames, 6) if self.sampled_frames else "",
            "frame_hash_unique_rate": round(len(self.frame_hashes) / self.sampled_frames, 6) if self.sampled_frames else "",
            "mean_frame_std": round(self.frame_std_sum / self.sampled_frames, 6) if self.sampled_frames else "",
            "mean_sample_frame_delta": round(self.frame_delta_sum / self.frame_delta_count, 6) if self.frame_delta_count else "",
            "player_samples": self.player_samples,
            "unique_player_positions": len(self.player_positions),
            "player_position_cap_reached": self.player_position_cap_reached,
            "player_grid_coverage": round(player_grid_coverage, 6),
        }

    def compute_player_grid_coverage(self) -> float:
        if not self.player_positions:
            return 0.0
        arr = np.asarray(list(self.player_positions), dtype=np.float64)
        x = arr[:, 0]
        y = arr[:, 1]
        xr = max(float(x.max() - x.min()), 1.0)
        yr = max(float(y.max() - y.min()), 1.0)
        gx = np.clip(((x - x.min()) / xr * self.player_grid_bins).astype(np.int64), 0, self.player_grid_bins - 1)
        gy = np.clip(((y - y.min()) / yr * self.player_grid_bins).astype(np.int64), 0, self.player_grid_bins - 1)
        cells = {(int(cx), int(cy)) for cx, cy in zip(gx.tolist(), gy.tolist())}
        return len(cells) / float(self.player_grid_bins ** 2)


def read_optional_array(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.load(path, mmap_mode="r")


def read_n_actions(path: Path) -> int | None:
    if not path.is_file():
        return None
    arr = np.asarray(np.load(path))
    if arr.size == 0:
        return None
    return int(arr.reshape(-1)[0])


def sample_indices(n: int, max_frames: int) -> np.ndarray:
    if n <= 0 or max_frames <= 0:
        return np.zeros(0, dtype=np.int64)
    count = min(n, max_frames)
    if count == n:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, count, dtype=np.int64))


def analyze_root(root: Path, dataset: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    env_stats: dict[tuple[str, str], DiversityStats] = {}
    group_stats: dict[tuple[str, str, str, str], DiversityStats] = {}

    for split, env, shard in iter_shards(root):
        metadata = load_json(shard / "metadata.json")
        profile = str(metadata.get("profile") or "unknown")
        level = str(metadata.get("level_index") or infer_level_index(metadata.get("level_file")))

        obs = np.load(shard / "obs.npy", mmap_mode="r")
        actions = np.load(shard / "action.npy", mmap_mode="r")
        n_actions = read_n_actions(shard / "n_actions.npy")
        restarted = read_optional_array(shard / "restarted.npy")
        player_x = read_optional_array(shard / "player_x.npy")
        player_y = read_optional_array(shard / "player_y.npy")

        env_key = (split, env)
        if env_key not in env_stats:
            env_stats[env_key] = DiversityStats(dataset=dataset, split=split, env=env)
        group_key = (split, env, profile, level)
        if group_key not in group_stats:
            group_stats[group_key] = DiversityStats(
                dataset=dataset, split=split, env=env, profile=profile, level=level
            )

        stats_list = (env_stats[env_key], group_stats[group_key])
        idx = sample_indices(int(obs.shape[0]), args.max_sampled_frames_per_shard)
        sampled = np.asarray(obs[idx]) if idx.size else np.zeros((0,), dtype=np.uint8)

        for stats in stats_list:
            stats.add_shard_meta(obs, metadata, env)
            stats.add_actions(actions, n_actions)
            stats.add_restarted(restarted)
            stats.add_player_positions(player_x, player_y, args.max_player_positions)
            stats.add_sampled_frames(sampled, args.hash_stride, args.max_sampled_frames_per_group)

    return (
        [stats.row() for stats in sorted(env_stats.values(), key=lambda s: (s.split, s.env))],
        [stats.row() for stats in sorted(group_stats.values(), key=lambda s: (s.split, s.env, s.profile, s.level))],
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def numeric(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def compare_rows(current: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_by_key = {(r["split"], r["env"]): r for r in baseline}
    out: list[dict[str, Any]] = []
    for cur in current:
        key = (cur["split"], cur["env"])
        old = base_by_key.get(key)
        if old is None:
            continue
        row = {
            "split": cur["split"],
            "env": cur["env"],
            "current_frames": cur["frames"],
            "baseline_frames": old["frames"],
        }
        for metric in (
            "action_entropy_bits",
            "action_coverage",
            "frame_hash_unique_rate",
            "mean_sample_frame_delta",
            "player_grid_coverage",
            "unique_player_positions",
        ):
            c = numeric(cur, metric)
            b = numeric(old, metric)
            row[f"current_{metric}"] = cur.get(metric, "")
            row[f"baseline_{metric}"] = old.get(metric, "")
            row[f"delta_{metric}"] = round(c - b, 6) if math.isfinite(c) and math.isfinite(b) else ""
        out.append(row)
    return out


def resize_frame(frame: np.ndarray, width: int) -> "Image.Image":
    from PIL import Image

    rgb = np.asarray(frame[..., :3], dtype=np.uint8)
    h, w = rgb.shape[:2]
    target_w = max(1, int(width))
    target_h = max(1, int(round(h * target_w / max(w, 1))))
    return Image.fromarray(rgb).resize((target_w, target_h), Image.NEAREST)


def collect_visual_frames(
    root: Path,
    group_by: str,
    frames_per_group: int,
    max_groups: int,
) -> dict[str, list[np.ndarray]]:
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for split, env, shard in iter_shards(root):
        if len(groups) >= max_groups and env not in groups:
            continue
        metadata = load_json(shard / "metadata.json")
        profile = str(metadata.get("profile") or "unknown")
        level = str(metadata.get("level_index") or infer_level_index(metadata.get("level_file")))
        if group_by == "profile_level":
            group = f"{split}__{env}__{profile}__lvl{level}"
        else:
            group = f"{split}__{env}"
        if len(groups[group]) >= frames_per_group:
            continue
        obs = np.load(shard / "obs.npy", mmap_mode="r")
        need = frames_per_group - len(groups[group])
        idx = sample_indices(int(obs.shape[0]), min(need, 4))
        for frame in np.asarray(obs[idx]):
            groups[group].append(np.asarray(frame))
            if len(groups[group]) >= frames_per_group:
                break
    return dict(groups)


def write_visuals(root: Path, out_dir: Path, args: argparse.Namespace) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("PIL is not installed; skipping visual sheets.")
        return

    visual_dir = out_dir / "visual_samples"
    visual_dir.mkdir(parents=True, exist_ok=True)
    groups = collect_visual_frames(
        root,
        args.visual_group,
        args.visual_frames_per_group,
        args.visual_max_groups,
    )
    for name, frames in sorted(groups.items()):
        if not frames:
            continue
        thumbs = [resize_frame(frame, args.visual_thumb_width) for frame in frames]
        cols = min(args.visual_cols, len(thumbs))
        rows = int(math.ceil(len(thumbs) / cols))
        label_h = 22
        w, h = thumbs[0].size
        sheet = Image.new("RGB", (cols * w, rows * (h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, thumb in enumerate(thumbs):
            x = (i % cols) * w
            y = (i // cols) * (h + label_h)
            draw.text((x + 4, y + 3), f"{i}", fill=(0, 0, 0))
            sheet.paste(thumb.convert("RGB"), (x, y + label_h))
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        sheet.save(visual_dir / f"{safe_name}_sheet.jpg", quality=90)

        gif_frames = thumbs[: args.visual_gif_frames]
        if gif_frames:
            first, rest = gif_frames[0], gif_frames[1:]
            first.save(
                visual_dir / f"{safe_name}.gif",
                save_all=True,
                append_images=rest,
                duration=max(1, int(1000 / args.visual_gif_fps)),
                loop=0,
            )
    print(f"Wrote visual samples to {visual_dir}")


def write_summary(path: Path, rows: list[dict[str, Any]], compare: list[dict[str, Any]] | None) -> None:
    total_frames = sum(int(r["frames"]) for r in rows)
    lines = [
        "# Transition Diversity Summary",
        "",
        f"Env rows: {len(rows)}",
        f"Total frames: {total_frames}",
        "",
        "## By Env",
        "",
        "| split | env | frames | profiles | levels | action H | act cov | hash unique | frame delta | player grid |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['split']} | {r['env']} | {r['frames']} | {r['profiles_seen']} | {r['levels_seen']} | "
            f"{r['action_entropy_bits']} | {r['action_coverage']} | {r['frame_hash_unique_rate']} | "
            f"{r['mean_sample_frame_delta']} | {r['player_grid_coverage']} |"
        )
    if compare:
        lines.extend([
            "",
            "## Current Minus Baseline",
            "",
            "| split | env | d action H | d act cov | d hash unique | d frame delta | d player grid |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for r in compare:
            lines.append(
                f"| {r['split']} | {r['env']} | {r['delta_action_entropy_bits']} | "
                f"{r['delta_action_coverage']} | {r['delta_frame_hash_unique_rate']} | "
                f"{r['delta_mean_sample_frame_delta']} | {r['delta_player_grid_coverage']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze diversity and validity of GVGAI transition shards.")
    parser.add_argument("--root", default="/hdd2/soyuj/transition_data", help="Current transition root.")
    parser.add_argument("--baseline-root", default="", help="Optional older transition root for comparison.")
    parser.add_argument("--out", default="", help="Output analysis directory. Defaults to <root>/analysis.")
    parser.add_argument("--label", default="current", help="Dataset label for --root.")
    parser.add_argument("--baseline-label", default="baseline", help="Dataset label for --baseline-root.")
    parser.add_argument("--max-sampled-frames-per-shard", type=int, default=8)
    parser.add_argument("--max-sampled-frames-per-group", type=int, default=5000)
    parser.add_argument("--max-player-positions", type=int, default=200000)
    parser.add_argument("--hash-stride", type=int, default=4)
    parser.add_argument("--write-visuals", action="store_true", help="Write contact sheets and GIFs for eye checks.")
    parser.add_argument("--visual-group", choices=("env", "profile_level"), default="env")
    parser.add_argument("--visual-frames-per-group", type=int, default=36)
    parser.add_argument("--visual-max-groups", type=int, default=200)
    parser.add_argument("--visual-thumb-width", type=int, default=180)
    parser.add_argument("--visual-cols", type=int, default=6)
    parser.add_argument("--visual-gif-frames", type=int, default=36)
    parser.add_argument("--visual-gif-fps", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_rows, group_rows = analyze_root(root, args.label, args)
    all_env_rows = list(env_rows)
    all_group_rows = list(group_rows)
    compare = None

    if args.baseline_root:
        baseline_root = Path(args.baseline_root).expanduser().resolve()
        baseline_env_rows, baseline_group_rows = analyze_root(baseline_root, args.baseline_label, args)
        compare = compare_rows(env_rows, baseline_env_rows)
        all_env_rows = baseline_env_rows + env_rows
        all_group_rows = baseline_group_rows + group_rows
        write_csv(out_dir / "diversity_compare.csv", compare)

    write_csv(out_dir / "diversity_by_env.csv", all_env_rows)
    write_csv(out_dir / "diversity_by_profile_level.csv", all_group_rows)
    write_summary(out_dir / "diversity_summary.md", env_rows, compare)

    if args.write_visuals:
        write_visuals(root, out_dir, args)

    print(f"Wrote {out_dir / 'diversity_by_env.csv'}")
    print(f"Wrote {out_dir / 'diversity_by_profile_level.csv'}")
    print(f"Wrote {out_dir / 'diversity_summary.md'}")
    if compare is not None:
        print(f"Wrote {out_dir / 'diversity_compare.csv'}")


if __name__ == "__main__":
    main()
