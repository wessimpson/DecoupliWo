"""Path resolution for games_world_model (per-game dirs, shared lvl*.txt)."""
from __future__ import annotations

import re
from os import path

NUM_WORLD_MODEL_LEVELS = 6  # lvl0 .. lvl5


def game_base_from_stem(vgdl_stem: str) -> str:
    if "_rules_" in vgdl_stem:
        return vgdl_stem.split("_rules_", 1)[0]
    return vgdl_stem


def is_legacy_world_model_package(name: str) -> bool:
    return bool(re.fullmatch(r".+_v\d+", name))


def resolve_gvgai_paths(base_dir: str, game: str, version: int) -> tuple[str, list[str]]:
    """
    Return (game_file, level_files) as absolute paths for the Java bridge.

    New layout: games_world_model/{base}/{stem}.txt + lvl0..lvl5.txt (shared).
    Legacy: games_world_model|games/{game}_v{version}/{game}_lvl*.txt
    """
    base = game_base_from_stem(game)
    wm_dir = path.join(base_dir, "games_world_model", base)
    game_file = path.join(wm_dir, f"{game}.txt")
    if path.isdir(wm_dir) and path.isfile(game_file):
        level_files = [
            path.realpath(path.join(wm_dir, f"lvl{i}.txt"))
            for i in range(NUM_WORLD_MODEL_LEVELS)
        ]
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
