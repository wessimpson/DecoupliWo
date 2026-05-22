from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PACKAGE_ROOT / "data" / "gvgai_rollouts"
DEFAULT_EXPORT_DIR = PACKAGE_ROOT / "data" / "hf_gvgai_dataset"


def read_json(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def iter_shards(data_root: Path):
	for metadata_path in sorted(data_root.glob("*/*/*/shard_*/metadata.json")):
		shard_dir = metadata_path.parent
		metadata = read_json(metadata_path)
		yield shard_dir, metadata


def relative_to_root(path: Path, root: Path) -> str:
	return path.resolve().relative_to(root.resolve()).as_posix()


def build_export(data_root: Path, export_dir: Path, repo_id: str | None, dataset_name: str) -> None:
	if export_dir.exists():
		shutil.rmtree(export_dir)
	export_dir.mkdir(parents=True, exist_ok=True)

	records: list[dict[str, Any]] = []
	for shard_dir, metadata in iter_shards(data_root):
		target = export_dir / "data" / relative_to_root(shard_dir, data_root)
		target.mkdir(parents=True, exist_ok=True)
		for file_path in sorted(shard_dir.iterdir()):
			if file_path.is_file():
				shutil.copy2(file_path, target / file_path.name)
		records.append({
			"split": metadata["split"],
			"game": metadata["game"],
			"variant": metadata["variant"],
			"env_key": metadata["env_key"],
			"shard_path": f"data/{relative_to_root(shard_dir, data_root)}",
			"num_rows": metadata["num_rows"],
			"obs_shape": metadata["obs_shape"],
			"rule_flags": metadata.get("rule_flags", {}),
			"schema_version": metadata["schema_version"],
		})

	(export_dir / "manifest.jsonl").write_text(
		"".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
		encoding="utf-8",
	)
	(export_dir / "dataset_info.json").write_text(
		json.dumps({
			"dataset_name": dataset_name,
			"repo_id": repo_id,
			"num_shards": len(records),
			"num_rows": sum(int(record["num_rows"]) for record in records),
			"splits": sorted({record["split"] for record in records}),
			"games": sorted({record["game"] for record in records}),
			"variants": sorted({record["variant"] for record in records}),
			"storage": "NumPy shards plus JSONL manifest",
		}, indent=2, sort_keys=True),
		encoding="utf-8",
	)
	(export_dir / ".gitattributes").write_text(
		"*.npy filter=lfs diff=lfs merge=lfs -text\n*.npz filter=lfs diff=lfs merge=lfs -text\n",
		encoding="utf-8",
	)
	(export_dir / "README.md").write_text(render_card(dataset_name, repo_id, records), encoding="utf-8")


def render_card(dataset_name: str, repo_id: str | None, records: list[dict[str, Any]]) -> str:
	total_rows = sum(int(record["num_rows"]) for record in records)
	games = sorted({record["game"] for record in records})
	variants = sorted({record["variant"] for record in records})
	repo_line = f"- Hub repo: `{repo_id}`\n" if repo_id else ""
	return f"""---
license: mit
task_categories:
- reinforcement-learning
- video-prediction
tags:
- gvgai
- world-model
- arcade-games
---

# {dataset_name}

Variant-aware GVGAI rollout dataset for world-model training.

{repo_line}- Storage schema: NumPy rollout shards plus `manifest.jsonl`
- Rows: {total_rows}
- Shards: {len(records)}
- Games: {", ".join(games) if games else "none"}
- Variants: {", ".join(variants) if variants else "none"}

Each shard directory contains `obs.npy`, `next_obs.npy`, `action.npy`, `reward.npy`,
episode boundary fields, level/seed fields, `rule_flags.npy`, and `metadata.json`.
The `rule_flags.npy` columns are ordered by `metadata.json["flag_names"]`.

Use `manifest.jsonl` as the stable index. Large `.npy` files are stored with Git LFS.
"""


def upload_with_hf_cli(export_dir: Path, repo_id: str, private: bool) -> None:
	from huggingface_hub import HfApi

	api = HfApi()
	api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
	api.upload_folder(folder_path=str(export_dir), repo_id=repo_id, repo_type="dataset")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Build or upload a Hugging Face dataset folder for GVGAI rollouts.")
	parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
	parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
	parser.add_argument("--dataset-name", default="GVGAI variant rollouts")
	parser.add_argument("--repo-id", default=None, help="Optional namespace/name for upload.")
	parser.add_argument("--upload", action="store_true")
	parser.add_argument("--private", action="store_true")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	build_export(args.data_root, args.export_dir, args.repo_id, args.dataset_name)
	if args.upload:
		if not args.repo_id:
			raise ValueError("--upload requires --repo-id namespace/name")
		upload_with_hf_cli(args.export_dir, args.repo_id, args.private)


if __name__ == "__main__":
	main()
