#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pong_common import ACTION_NAMES, ID_TO_EVENT, ID_TO_GAME, ID_TO_RULE, ID_TO_SOURCE, load_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a custom Pong transition dataset.")
    parser.add_argument("dataset", help="Dataset root containing train/val shards.")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--examples", type=int, default=5)
    return parser.parse_args()


def histogram(values: np.ndarray) -> dict[int, int]:
    keys, counts = np.unique(values, return_counts=True)
    return {int(key): int(count) for key, count in zip(keys, counts)}


def named_hist(values: np.ndarray, names: dict[int, str]) -> dict[str, int]:
    return {names.get(key, str(key)): count for key, count in histogram(values).items()}


def main() -> int:
    args = parse_args()
    root = pathlib.Path(args.dataset).expanduser().resolve()
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        print("metadata:")
        for key in ("collector", "policy", "episodes", "steps_per_episode", "seed", "counterfactual_base_mode"):
            if key in metadata:
                print(f"  {key}: {metadata[key]}")

    data = load_shards(root, args.split)
    n = int(data["action"].shape[0])
    print(f"split: {args.split}")
    print(f"rows: {n}")
    print(f"state shape: {data['state'].shape}")
    print(f"next_state shape: {data['next_state'].shape}")
    if "object_slots" in data:
        print(f"object_slots shape: {data['object_slots'].shape}")
        print(f"object_mask active mean: {data['object_mask'].sum(axis=1).mean():.2f}")
    print(f"reward mean/std/min/max: {data['reward'].mean():.4f} {data['reward'].std():.4f} {data['reward'].min():.4f} {data['reward'].max():.4f}")
    if "game_id" in data:
        print(f"games: {named_hist(data['game_id'], ID_TO_GAME)}")
    print(f"rules: {named_hist(data['rule_id'], ID_TO_RULE)}")
    print(f"actions: {named_hist(data['action'], ACTION_NAMES)}")
    print(f"events: {named_hist(data['event_id'], ID_TO_EVENT)}")
    if "source_id" in data:
        print(f"sources: {named_hist(data['source_id'], ID_TO_SOURCE)}")
    print(f"terminated: {int(data['terminated'].sum())}")
    print(f"truncated: {int(data['truncated'].sum())}")

    rng = np.random.default_rng(0)
    count = min(int(args.examples), n)
    if count:
        print("examples:")
        for idx in rng.choice(n, size=count, replace=False):
            state = np.round(data["state"][idx], 3).tolist()
            next_state = np.round(data["next_state"][idx], 3).tolist()
            rule = ID_TO_RULE.get(int(data["rule_id"][idx]), str(int(data["rule_id"][idx])))
            action = ACTION_NAMES.get(int(data["action"][idx]), str(int(data["action"][idx])))
            event = ID_TO_EVENT.get(int(data["event_id"][idx]), str(int(data["event_id"][idx])))
            source = ID_TO_SOURCE.get(int(data["source_id"][idx]), str(int(data["source_id"][idx]))) if "source_id" in data else "unknown"
            print(f"  idx={int(idx)} source={source} rule={rule} action={action} event={event} reward={float(data['reward'][idx]):.2f}")
            print(f"    state={state}")
            print(f"    next ={next_state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
