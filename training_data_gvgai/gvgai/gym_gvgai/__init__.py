from gymnasium.envs.registration import register, make
import os
import re
import sys
import subprocess

dir = os.path.dirname(__file__)


def marker_version_ok(marker_path: str) -> bool:
    with open(marker_path, encoding="utf-8") as f:
        first = f.readline().strip()
    return first in (
        "v2_data_collection_fidelity",
        "v3_game_folders_shared_levels",
        "v4_data_collection_fidelity",
        "v5_data_collection_rules_classic_sprites",
        "v6_data_collection_rules_classic_sprite_visuals",
        "v7_exact_data_collection_vgdl",
        "v8_aliens_variants_data_collection_vgdl",
        "v9_square_size_8",
        "v10_exact_ground_truth_data_collection",
        "v11_square_size_8",
        "v12_grid_10x10_square_8",
        "v13_grid_15x15_square_8",
        "v14_world_model_variants_mixed_grids",
        "v15_world_model_variants_15x15",
    )


def _ensure_world_model_games() -> None:
    wm = os.path.join(dir, "envs", "games_world_model")
    if not os.path.isdir(wm):
        return
    marker = os.path.join(wm, ".built_from_data_collection")
    try:
        if os.path.isfile(marker) and marker_version_ok(marker):
            return
    except OSError:
        pass
    for script in (
        os.path.join(dir, "..", "..", "..", "GVGAI_jpype", "scripts", "build_world_model_games.py"),
        os.path.join(dir, "..", "scripts", "build_world_model_games.py"),
    ):
        if os.path.isfile(script):
            subprocess.run([sys.executable, script], check=False)
            return


_ensure_world_model_games()


def _parse_game_folder(folder: str) -> tuple[str, int] | None:
    if "_v" not in folder:
        return None
    name, _, ver = folder.rpartition("_v")
    if not ver.isdigit():
        return None
    return name, int(ver)


def _register_legacy_packages(games_path: str, base_dir: str) -> None:
    if not os.path.isdir(games_path):
        return
    for game in os.listdir(games_path):
        game_path = os.path.join(games_path, game)
        if not os.path.isdir(game_path):
            continue
        parsed = _parse_game_folder(game)
        if parsed is None:
            continue
        name, version = parsed
        lvls = len([lvl for lvl in os.listdir(game_path) if "lvl" in lvl])
        for lvl in range(lvls):
            register(
                id=f"gvgai-{name}-lvl{lvl}-v{version}",
                entry_point="gym_gvgai.envs.gvgai_env_jpype:GVGAI_Env_JPype",
                kwargs={
                    "game": name,
                    "level": lvl,
                    "version": version,
                    "base_dir": base_dir,
                },
                max_episode_steps=2000,
            )


def _register_world_model_game_dirs(wm_path: str, base_dir: str) -> None:
    if not os.path.isdir(wm_path):
        return
    for base_game in sorted(os.listdir(wm_path)):
        if base_game.startswith("."):
            continue
        game_dir = os.path.join(wm_path, base_game)
        if not os.path.isdir(game_dir):
            continue
        if _parse_game_folder(base_game) is not None:
            continue

        lvl_files = sorted(
            (f for f in os.listdir(game_dir) if re.fullmatch(r"lvl\d+\.txt", f)),
            key=lambda f: int(re.search(r"\d+", f).group()),
        )
        if not lvl_files:
            continue
        n_lvls = len(lvl_files)

        for fname in sorted(os.listdir(game_dir)):
            if not fname.endswith(".txt") or fname.startswith("lvl"):
                continue
            stem = fname[:-4]
            for lvl in range(n_lvls):
                register(
                    id=f"gvgai-{stem}-lvl{lvl}-v0",
                    entry_point="gym_gvgai.envs.gvgai_env_jpype:GVGAI_Env_JPype",
                    kwargs={
                        "game": stem,
                        "level": lvl,
                        "version": 0,
                        "base_dir": base_dir,
                    },
                    max_episode_steps=2000,
                )


envs_dir = os.path.join(dir, "envs")
_register_legacy_packages(os.path.join(envs_dir, "games"), envs_dir)
_register_world_model_game_dirs(os.path.join(envs_dir, "games_world_model"), envs_dir)
