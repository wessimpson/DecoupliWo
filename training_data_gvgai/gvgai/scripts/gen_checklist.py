"""Regenerate games_world_model/CHECKLIST.md from on-disk VGDL variants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "gym_gvgai" / "envs" / "games_world_model"
OUT = ROOT / "CHECKLIST.md"

IKARUGA_ORDER = [
    "big_explosion_3rad",
    "big_shot_1x",
    "big_shot_2x",
    "big_shot_4x",
    "multishot_2",
    "multishot_3",
    "multishot_5",
    "pierce_shot",
    "quick_dash_2tile",
    "quick_dash_3tile",
    "quick_dash_5tile",
    "shield_reflect",
]


def sort_variants(tags: list[str]) -> list[str]:
    ik = [t for t in IKARUGA_ORDER if t in tags]
    rest = sorted(t for t in tags if t not in IKARUGA_ORDER)
    return ik + rest


def collect_games() -> dict[str, dict]:
    games: dict[str, dict] = {}
    for game_dir in sorted(ROOT.iterdir()):
        if not game_dir.is_dir() or game_dir.name.startswith("."):
            continue
        game = game_dir.name
        tags = [
            p.stem[len(f"{game}_rules_") :]
            for p in sorted(game_dir.glob(f"{game}_rules_*.txt"))
        ]
        has_base = (game_dir / f"{game}.txt").is_file()
        if not has_base and not tags:
            continue
        n_lvls = len(list(game_dir.glob("lvl*.txt")))
        games[game] = {
            "tags": sort_variants(tags),
            "has_base": has_base,
            "levels": n_lvls,
        }
    return games


def main() -> None:
    games = collect_games()
    total_variants = sum(
        (1 if info["has_base"] else 0) + len(info["tags"]) for info in games.values()
    )

    lines: list[str] = [
        "# games_world_model checklist",
        "",
        "Track verification of every world-model game and rule variant.",
        "",
        "## Summary",
        "",
        "| Game | Variants | Levels |",
        "|---|---:|---:|",
    ]
    for game, info in games.items():
        n = (1 if info["has_base"] else 0) + len(info["tags"])
        lines.append(f"| {game} | {n} | {info['levels']} |")
    lines += [
        f"| **Total** | **{total_variants}** | |",
        "",
        "## How to verify",
        "",
        "Mark **rule working** when the variant mechanic behaves as intended in play.",
        "Mark **agent working** when MCTS/random can run episodes without Java errors or crashes.",
        "",
        "```powershell",
        "python training_data_gvgai/run_mcts.py --env <game> --rules <tag> --level 0 --show",
        "python training_data_gvgai/run_random_action.py --env <game> --rules <tag> --level 0 --show",
        "# base game: omit --rules",
        "```",
        "",
        "### Rule families",
        "",
        "| Family | Tags |",
        "|---|---|",
        "| **Ikaruga-style** | `big_explosion_3rad`, `big_shot_*`, `multishot_2/3/5`, `pierce_shot`, `quick_dash_*`, `shield_reflect` |",
        "| **Shooter extras** | `enemy_explode_*rad`, `enemy_multishot`, `multishot` (spread), `ricochet`, `shoot_walls`, `split_orthogonal`, `two_hit_color` |",
        "| **Other** | `enemy_speed_2x`, `car_speed_2x`, `oil_slowdown`, `ghost_*`, `wall_on_death`, etc. |",
        "",
        "---",
        "",
    ]

    for game, info in games.items():
        lines.append(f"## {game}")
        lines.append("")
        if info["levels"]:
            lines.append(
                f"*{info['levels']} shared level(s): `lvl0` … `lvl{info['levels'] - 1}`*"
            )
        else:
            lines.append("*No level files found.*")
        lines.append("")
        lines.append("| Variant | `--rules` tag | rule working | agent working |")
        lines.append("|---|---|:---:|:---:|")

        if info["has_base"]:
            lines.append(f"| `{game}` | *(omit)* | [ ] | [ ] |")

        ik_tags = [t for t in info["tags"] if t in IKARUGA_ORDER]
        other_tags = [t for t in info["tags"] if t not in IKARUGA_ORDER]

        if ik_tags:
            for tag in ik_tags:
                lines.append(f"| `{game}_rules_{tag}` | `{tag}` | [ ] | [ ] |")

        if other_tags:
            if ik_tags:
                lines.append("| *game-specific* | | | |")
            for tag in other_tags:
                lines.append(f"| `{game}_rules_{tag}` | `{tag}` | [ ] | [ ] |")

        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({total_variants} variants across {len(games)} games)")


if __name__ == "__main__":
    main()
