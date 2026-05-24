"""Path resolution for games_world_model (per-game dirs, shared lvl*.txt)."""
from __future__ import annotations

import os
import re
from os import path


def game_base_from_stem(vgdl_stem: str) -> str:
    if "_rules_" in vgdl_stem:
        return vgdl_stem.split("_rules_", 1)[0]
    return vgdl_stem


def is_legacy_world_model_package(name: str) -> bool:
    return bool(re.fullmatch(r".+_v\d+", name))


def _world_model_level_files(wm_dir: str) -> list[str]:
    """Discover lvl0.txt, lvl1.txt, … that actually exist (sparse counts OK)."""
    if not path.isdir(wm_dir):
        return []
    paths = [
        path.join(wm_dir, name)
        for name in sorted(os.listdir(wm_dir))
        if re.fullmatch(r"lvl\d+\.txt", name)
    ]
    paths.sort(key=lambda p: int(re.search(r"lvl(\d+)", path.basename(p)).group(1)))
    return [path.realpath(p) for p in paths]


def resolve_gvgai_paths(base_dir: str, game: str, version: int) -> tuple[str, list[str]]:
    """
    Return (game_file, level_files) as absolute paths for the Java bridge.

    New layout: games_world_model/{base}/{stem}.txt + lvl*.txt (shared).
    Legacy: games_world_model|games/{game}_v{version}/{game}_lvl*.txt
    """
    base = game_base_from_stem(game)
    wm_dir = path.join(base_dir, "games_world_model", base)
    game_file = path.join(wm_dir, f"{game}.txt")
    if path.isdir(wm_dir):
        if not path.isfile(game_file):
            available = sorted(
                name[len(f"{base}_rules_") :]
                for name in os.listdir(wm_dir)
                if name.startswith(f"{base}_rules_") and name.endswith(".txt")
            )
            hint = f" Available {base} rules: {', '.join(available) or '(none)'}"
            raise FileNotFoundError(f"GVGAI game file not found: {game_file}.{hint}")
        level_files = _world_model_level_files(wm_dir)
        if not level_files:
            raise FileNotFoundError(f"No lvl*.txt files in {wm_dir}")
        level_files.append("")
        return path.realpath(game_file), level_files

    legacy_dir = None
    for sub in ("games_world_model", "games"):
        candidate = path.join(base_dir, sub, f"{game}_v{version}")
        if path.isdir(candidate):
            legacy_dir = candidate
            break
    if legacy_dir is None:
        legacy_dir = path.join(base_dir, "games", f"{game}_v{version}")

    game_file = path.realpath(path.join(legacy_dir, f"{game}.txt"))
    level_files = [
        path.realpath(path.join(legacy_dir, f"{game}_lvl{i}.txt"))
        for i in range(5)
    ]
    level_files.append("")
    return game_file, level_files
