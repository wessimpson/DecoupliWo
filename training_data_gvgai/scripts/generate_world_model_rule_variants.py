from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GAMES_ROOT = ROOT / "training_data_gvgai" / "gvgai" / "gym_gvgai" / "envs" / "games"
WM_ROOT = ROOT / "training_data_gvgai" / "gvgai" / "gym_gvgai" / "envs" / "games_world_model"
CATALOG_PATH = ROOT / "training_data_gvgai" / "data" / "gvgai_variant_catalog.world_model.json"
MARKER_PATH = WM_ROOT / ".built_from_data_collection"
MARKER_VERSION = "v14_world_model_variants_mixed_grids"


LEGACY_ONLY = ("ikaruga", "seaquest", "missilecommand", "frogs", "pacman", "roadfighter", "sheriff")
BASES = ("defender", "jaws", "zelda", *LEGACY_ONLY)

QUICK_DASH = {
    "defender": ("avatar", 1.0),
    "jaws": ("avatar", 1.0),
    "zelda": ("avatar", 1.0),
    "ikaruga": ("avatar", 1.0),
    "seaquest": ("avatar", 1.0),
    "frogs": ("avatar", 1.0),
    "pacman": ("pacman", 0.5),
    "roadfighter": ("avatar", 0.5),
    "sheriff": ("avatar", 1.0),
}

PROJECTILES = {
    "defender": ("avatar", "sam"),
    "ikaruga": ("whiteAvatar", "whiteBullet"),
    "seaquest": ("avatar", "torpedo"),
    "missilecommand": ("avatar", "explosion"),
    "sheriff": ("avatar", "bullet"),
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_square_size_8(text: str) -> str:
    first, *rest = text.splitlines()
    if "square_size=" in first:
        first = re.sub(r"square_size=\d+", "square_size=8", first, count=1)
    else:
        first = first.replace("BasicGame", "BasicGame square_size=8", 1)
    return "\n".join([first, *rest])


def copy_legacy_games() -> None:
    for game in LEGACY_ONLY:
        src_dir = GAMES_ROOT / f"{game}_v0"
        dst_dir = WM_ROOT / game
        dst_dir.mkdir(parents=True, exist_ok=True)
        write_text(dst_dir / f"{game}.txt", ensure_square_size_8(read_text(src_dir / f"{game}.txt")))
        for i in range(5):
            write_text(dst_dir / f"lvl{i}.txt", read_text(src_dir / f"{game}_lvl{i}.txt"))
        write_text(dst_dir / "lvl5.txt", read_text(src_dir / f"{game}_lvl0.txt"))


def base_text(game: str) -> str:
    return read_text(WM_ROOT / game / f"{game}.txt")


def section_bounds(text: str, name: str) -> tuple[int, int]:
    header = re.search(rf"(?m)^\s*{re.escape(name)}\s*$", text)
    if not header:
        raise ValueError(f"Missing section {name}")
    start = header.start()
    next_header = re.search(r"(?m)^\s*(SpriteSet|LevelMapping|InteractionSet|TerminationSet)\s*$", text[header.end() :])
    if not next_header:
        return start, len(text)
    return start, header.end() + next_header.start()


def replace_first_line(text: str, prefix: str, new_line: str) -> str:
    return re.sub(rf"(?m)^(?P<indent>\s*){re.escape(prefix)}.*$", lambda m: m.group("indent") + new_line, text, count=1)


def scale_quick_dash(text: str, sprite: str, multiplier: int, fallback_speed: float) -> str:
    pattern = re.compile(rf"(?m)^(?P<line>\s*{re.escape(sprite)}\s*>\s*.*)$")
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Missing sprite line for {sprite}")
    line = match.group("line")
    speed_match = re.search(r"speed=([0-9.]+)", line)
    speed = float(speed_match.group(1)) if speed_match else fallback_speed
    new_speed = speed * multiplier
    if speed_match:
        new_line = re.sub(r"speed=[0-9.]+", f"speed={new_speed:.4g}", line, count=1)
    else:
        new_line = line + f" speed={new_speed:.4g}"
    return text[: match.start("line")] + new_line + text[match.end("line") :]


def scale_shrinkfactor(text: str, sprite: str, multiplier: float) -> str:
    pattern = re.compile(rf"(?m)^(?P<line>\s*{re.escape(sprite)}\s*>\s*.*)$")
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Missing projectile line for {sprite}")
    line = match.group("line")
    shrink_match = re.search(r"shrinkfactor=([0-9.]+)", line)
    shrink = float(shrink_match.group(1)) if shrink_match else 1.0
    new_line = re.sub(r"shrinkfactor=[0-9.]+", f"shrinkfactor={shrink * multiplier:.4g}", line, count=1) if shrink_match else line + f" shrinkfactor={shrink * multiplier:.4g}"
    return text[: match.start("line")] + new_line + text[match.end("line") :]


def replace_in_interaction(text: str, old: str, new: str) -> str:
    start, end = section_bounds(text, "InteractionSet")
    block = text[start:end].replace(old, new)
    return text[:start] + block + text[end:]


def add_sprite_before_levels(text: str, line: str, sprite_name: str) -> str:
    if re.search(rf"(?m)^\s*{re.escape(sprite_name)}\s*>", text):
        return text
    match = re.search(r"(?m)^\s*LevelMapping\s*$", text)
    if not match:
        raise ValueError("Missing LevelMapping section")
    return text[: match.start()] + line.rstrip() + "\n\n" + text[match.start() :]


def clone_projectile_variant(text: str, avatar_sprite: str, projectile_sprite: str, count: int) -> str:
    names = [f"{projectile_sprite}{i}" for i in range(1, count + 1)]
    avatar_pattern = re.compile(rf"(?m)^(?P<line>\s*{re.escape(avatar_sprite)}\s*>\s*.*)$")
    avatar_match = avatar_pattern.search(text)
    if not avatar_match:
        raise ValueError(f"Missing avatar line for {avatar_sprite}")
    avatar_line = avatar_match.group("line")
    avatar_line = re.sub(rf"stype={re.escape(projectile_sprite)}\b", f"stype={','.join(names)}", avatar_line)
    if "fireAllWeapons=True" not in avatar_line:
        avatar_line += " fireAllWeapons=True spreadPixels=0"
    text = text[: avatar_match.start("line")] + avatar_line + text[avatar_match.end("line") :]

    proj_pattern = re.compile(rf"(?m)^(?P<indent>\s*){re.escape(projectile_sprite)}\s*>\s*(?P<body>.*)$")
    proj_match = proj_pattern.search(text)
    if not proj_match:
        raise ValueError(f"Missing projectile line for {projectile_sprite}")
    indent = proj_match.group("indent")
    body = proj_match.group("body")
    new_block = "\n".join(f"{indent}{name} > {body}" for name in names)
    text = text[: proj_match.start()] + new_block + text[proj_match.end() :]
    start, end = section_bounds(text, "InteractionSet")
    text = text[:start] + text[start:end].replace(projectile_sprite, " ".join(names)) + text[end:]
    return text


def write_variant(game: str, stem: str, text: str) -> None:
    write_text(WM_ROOT / game / f"{stem}.txt", ensure_square_size_8(text))


def all_game_dirs() -> list[Path]:
    return sorted(p for p in WM_ROOT.iterdir() if p.is_dir() and not p.name.startswith("."))


def write_catalog() -> None:
    games: dict[str, object] = {}
    for game_dir in all_game_dirs():
        base = game_dir.name
        level_count = len(list(game_dir.glob("lvl*.txt")))
        variants: dict[str, object] = {
            "default": {
                "description": "Base world-model GVGAI rules.",
                "env_id": f"gvgai-{base}-lvl{{level}}-v0",
                "levels": list(range(level_count)),
                "rule_flags": {},
            },
        }
        for vgdl_path in sorted(game_dir.glob(f"{base}_rules_*.txt")):
            tag = vgdl_path.stem.split("_rules_", 1)[1]
            variants[tag] = {
                "description": f"{base} with rule tag {tag}.",
                "env_id": f"gvgai-{vgdl_path.stem}-lvl{{level}}-v0",
                "levels": list(range(level_count)),
                "rule_flags": {tag: 1},
            }
        games[base] = {
            "levels": list(range(level_count)),
            "variants": variants,
        }

    payload = {
        "defaults": {"levels": [0, 1, 2, 3, 4, 5]},
        "games": games,
        "metadata": {
            "games_world_model_root": "training_data_gvgai/gvgai/gym_gvgai/envs/games_world_model",
            "marker_version": MARKER_VERSION,
            "source": "training_data_gvgai/scripts/generate_world_model_rule_variants.py",
        },
    }
    write_json(CATALOG_PATH, payload)


def write_marker() -> None:
    game_dirs = all_game_dirs()
    total_games = len(game_dirs)
    total_variants = sum(len(list(game_dir.glob(f"{game_dir.name}_rules_*.txt"))) for game_dir in game_dirs)
    lines = [
        MARKER_VERSION,
        f"games={total_games}",
        f"variants={total_variants}",
        "levels=lvl0..lvl5",
        "grid=mixed",
        "square_size=8",
        "source=training_data_gvgai/scripts/generate_world_model_rule_variants.py",
    ]
    write_text(MARKER_PATH, "\n".join(lines))


def quick_dash_variants() -> None:
    for game, (sprite, fallback_speed) in QUICK_DASH.items():
        base = base_text(game)
        for mult in (2, 3, 5):
            write_variant(game, f"{game}_rules_quick_dash_{mult}tile", scale_quick_dash(base, sprite, mult, fallback_speed))


def big_shot_variants() -> None:
    for game in ("defender", "ikaruga", "seaquest", "missilecommand", "sheriff"):
        _, projectile = PROJECTILES[game]
        base = base_text(game)
        for mult in (1, 2, 4):
            write_variant(game, f"{game}_rules_big_shot_{mult}x", scale_shrinkfactor(base, projectile, mult))


def multishot_variants() -> None:
    for game in ("defender", "ikaruga", "seaquest", "missilecommand", "sheriff"):
        avatar, projectile = PROJECTILES[game]
        base = base_text(game)
        for count in (2, 3, 5):
            write_variant(game, f"{game}_rules_multishot_{count}", clone_projectile_variant(base, avatar, projectile, count))


def pierce_variants() -> None:
    mappings = {
        "defender": ("alien  sam   > killBoth scoreChange=2", "alien  sam   > killSprite scoreChange=2"),
        "ikaruga": ("blackAlien  blackBullet > killBoth scoreChange=1", "blackAlien  blackBullet > killSprite scoreChange=1"),
        "seaquest": ("fish torpedo > killBoth scoreChange=1", "fish torpedo > killSprite scoreChange=1"),
        "missilecommand": ("incoming explosion > killSprite scoreChange=2", "incoming explosion > transformTo stype=explosion killSecond=True scoreChange=2"),
        "sheriff": ("bandit bullet > killBoth scoreChange=1", "bandit bullet > killSprite scoreChange=1"),
    }
    for game, (old, new) in mappings.items():
        write_variant(game, f"{game}_rules_pierce_shot", replace_in_interaction(base_text(game), old, new))


def wall_on_death_variants() -> None:
    sheriff = add_sprite_before_levels(base_text("sheriff"), "        wall > Immovable autotiling=True img=oryx/dirtwall", "wall")
    sheriff = replace_in_interaction(sheriff, "bandit bullet > killBoth scoreChange=1", "bandit bullet > transformTo stype=wall killSecond=True scoreChange=1")
    write_variant("sheriff", "sheriff_rules_wall_on_death", sheriff)

    zelda = add_sprite_before_levels(base_text("zelda"), "      wall > Immovable autotiling=true img=oryx/wall3", "wall")
    zelda = replace_in_interaction(zelda, "enemy sword > killSprite scoreChange=2", "enemy sword > transformTo stype=wall killSecond=True scoreChange=2")
    write_variant("zelda", "zelda_rules_wall_on_death", zelda)

    pacman = add_sprite_before_levels(base_text("pacman"), "        wall > Immovable img=oryx/wall3 autotiling=True", "wall")
    pacman = replace_in_interaction(pacman, "ghost powered > killSprite scoreChange=40", "ghost powered > transformTo stype=wall scoreChange=40")
    write_variant("pacman", "pacman_rules_wall_on_death", pacman)


def speed_variants() -> None:
    def set_speed(game: str, sprite: str, speed: float) -> str:
        text = base_text(game)
        pattern = re.compile(rf"(?m)^(?P<line>\s*{re.escape(sprite)}\s*>\s*.*)$")
        match = pattern.search(text)
        if not match:
            raise ValueError(f"Missing sprite {sprite} in {game}")
        line = match.group("line")
        new_line = re.sub(r"speed=[0-9.]+", f"speed={speed:.4g}", line, count=1) if "speed=" in line else line + f" speed={speed:.4g}"
        return text[: match.start("line")] + new_line + text[match.end("line") :]

    write_variant("defender", "defender_rules_enemy_speed_2x", set_speed("defender", "alien", 1.2))

    jaws = base_text("jaws")
    for sprite, speed in (("shark", 0.4), ("whale", 0.4), ("piranha", 0.4)):
        pattern = re.compile(rf"(?m)^(?P<line>\s*{re.escape(sprite)}\s*>\s*.*)$")
        match = pattern.search(jaws)
        if match:
            line = match.group("line")
            new_line = re.sub(r"speed=[0-9.]+", f"speed={speed:.4g}", line, count=1) if "speed=" in line else line + f" speed={speed:.4g}"
            jaws = jaws[: match.start("line")] + new_line + jaws[match.end("line") :]
    write_variant("jaws", "jaws_rules_enemy_speed_2x", jaws)

    zelda = base_text("zelda")
    zelda = replace_first_line(zelda, "monsterQuick", "monsterQuick > RandomNPC cooldown=1 cons=6 img=oryx/bat1")
    zelda = replace_first_line(zelda, "monsterNormal", "monsterNormal > RandomNPC cooldown=2 cons=8 img=oryx/spider2")
    zelda = replace_first_line(zelda, "monsterSlow", "monsterSlow > RandomNPC cooldown=4 cons=12 img=oryx/scorpion1")
    write_variant("zelda", "zelda_rules_enemy_speed_2x", zelda)

    seaquest = base_text("seaquest")
    for sprite, speed in (("shark", 0.5), ("whale", 0.2), ("pirana", 0.5)):
        pattern = re.compile(rf"(?m)^(?P<line>\s*{re.escape(sprite)}\s*>\s*.*)$")
        match = pattern.search(seaquest)
        if match:
            line = match.group("line")
            new_line = re.sub(r"speed=[0-9.]+", f"speed={speed:.4g}", line, count=1) if "speed=" in line else line + f" speed={speed:.4g}"
            seaquest = seaquest[: match.start("line")] + new_line + seaquest[match.end("line") :]
    write_variant("seaquest", "seaquest_rules_enemy_speed_2x", seaquest)

    frogs = base_text("frogs")
    for sprite, speed in (("fastRtruck", 0.4), ("slowRtruck", 0.2), ("fastLtruck", 0.4), ("slowLtruck", 0.2)):
        frogs = re.sub(rf"(?m)^(\s*{re.escape(sprite)}\s*>\s*.*?speed=)[0-9.]+", rf"\g<1>{speed:.4g}", frogs, count=1)
    write_variant("frogs", "frogs_rules_car_speed_2x", frogs)

    road = base_text("roadfighter")
    for sprite, speed in (("carSlow", 0.5), ("carFast", 1.0)):
        road = re.sub(rf"(?m)^(\s*{re.escape(sprite)}\s*>\s*.*?speed=)[0-9.]+", rf"\g<1>{speed:.4g}", road, count=1)
    write_variant("roadfighter", "roadfighter_rules_car_speed_2x", road)

    pac = base_text("pacman")
    pac = replace_first_line(pac, "redOk", "redOk > RandomPathAltChaser stype1=hungry stype2=powered cooldown=2 img=oryx/ghost3 cons=4")
    pac = replace_first_line(pac, "blueOk", "blueOk > RandomPathAltChaser stype1=hungry stype2=powered cooldown=2 img=oryx/ghost4 cons=4")
    pac = replace_first_line(pac, "pinkOk", "pinkOk > RandomPathAltChaser stype1=hungry stype2=powered cooldown=2 img=oryx/ghost5 cons=4")
    pac = replace_first_line(pac, "orangeOk", "orangeOk > RandomPathAltChaser stype1=hungry stype2=powered cooldown=2 img=oryx/ghost6 cons=4")
    pac = replace_first_line(pac, "redSc", "redSc > Fleeing stype=pacman maxDistance=500 cooldown=1 img=oryx/ghost1")
    pac = replace_first_line(pac, "blueSc", "blueSc > Fleeing stype=pacman maxDistance=500 cooldown=1 img=oryx/ghost1")
    pac = replace_first_line(pac, "pinkSc", "pinkSc > Fleeing stype=pacman maxDistance=500 cooldown=1 img=oryx/ghost1")
    pac = replace_first_line(pac, "orangeSc", "orangeSc > Fleeing stype=pacman maxDistance=500 cooldown=1 img=oryx/ghost1")
    write_variant("pacman", "pacman_rules_ghost_speed_2x", pac)


def shield_reflect_variants() -> None:
    defender = base_text("defender")
    defender = replace_in_interaction(defender, "avatar bomb  > killSprite scoreChange=-1", "avatar bomb  > stepBack")
    defender = replace_in_interaction(defender, "missile EOS city > killSprite", "missile EOS city > killSprite\n\n        bomb avatar > reverseDirection\n        alien bomb > killBoth scoreChange=1")
    write_variant("defender", "defender_rules_shield_reflect", defender)

    ikaruga = base_text("ikaruga")
    ikaruga = replace_in_interaction(ikaruga, "avatar      bomb        > killBoth scoreChange=-1", "avatar      bomb        > stepBack\n        bomb        avatar      > reverseDirection")
    write_variant("ikaruga", "ikaruga_rules_shield_reflect", ikaruga)

    seaquest = base_text("seaquest")
    seaquest = replace_in_interaction(seaquest, "avatar fish  > killSprite", "avatar shark > killSprite\n        avatar whale > killSprite\n        avatar pirana > stepBack\n        pirana avatar > reverseDirection\n        fish pirana > killBoth scoreChange=1")
    write_variant("seaquest", "seaquest_rules_shield_reflect", seaquest)


def explosion_variants() -> None:
    zelda = add_sprite_before_levels(base_text("zelda"), "    explosion > Flicker limit=5 singleton=True img=oryx/circleEffect1 shrinkfactor=3", "explosion")
    zelda = replace_in_interaction(zelda, "enemy sword > killSprite scoreChange=2", "enemy sword > transformTo stype=explosion killSecond=True scoreChange=2\n    enemy explosion > killSprite")
    write_variant("zelda", "zelda_rules_big_explosion_3rad", zelda)

    ikaruga = add_sprite_before_levels(base_text("ikaruga"), "    explosion > Flicker limit=5 singleton=True img=oryx/circleEffect1 shrinkfactor=3", "explosion")
    ikaruga = replace_in_interaction(ikaruga, "blackAlien  blackBullet > killBoth scoreChange=1", "blackAlien  blackBullet > transformTo stype=explosion killSecond=True scoreChange=1\n        whiteAlien  whiteBullet > transformTo stype=explosion killSecond=True scoreChange=1\n        alien explosion > killSprite")
    write_variant("ikaruga", "ikaruga_rules_big_explosion_3rad", ikaruga)

    sheriff = add_sprite_before_levels(base_text("sheriff"), "    explosion > Flicker limit=5 singleton=True img=oryx/circleEffect1 shrinkfactor=3", "explosion")
    sheriff = replace_in_interaction(sheriff, "bandit bullet > killBoth scoreChange=1", "bandit bullet > transformTo stype=explosion killSecond=True scoreChange=1\n        bandit explosion > killSprite\n        missile explosion > killSprite")
    write_variant("sheriff", "sheriff_rules_big_explosion_3rad", sheriff)

    write_variant("missilecommand", "missilecommand_rules_big_explosion_3rad", scale_shrinkfactor(base_text("missilecommand"), "explosion", 3))
    write_variant("missilecommand", "missilecommand_rules_big_explosion_5rad", scale_shrinkfactor(base_text("missilecommand"), "explosion", 5))


def pacman_freeze_variant() -> None:
    pac = base_text("pacman")
    marker = (
        "                orange > singleton=True\n"
        "                    orangeOk > RandomPathAltChaser stype1=hungry stype2=powered cooldown=4 img=oryx/ghost6 cons=4\n"
        "                    orangeSc > Fleeing stype=pacman maxDistance=500 cooldown=2 img=oryx/ghost1\n"
    )
    insert = (
        marker +
        "                redFreeze > RandomPathAltChaser stype1=hungry stype2=powered cooldown=999 speed=0 img=oryx/ghost1 cons=4\n"
        "                blueFreeze > RandomPathAltChaser stype1=hungry stype2=powered cooldown=999 speed=0 img=oryx/ghost1 cons=4\n"
        "                pinkFreeze > RandomPathAltChaser stype1=hungry stype2=powered cooldown=999 speed=0 img=oryx/ghost1 cons=4\n"
        "                orangeFreeze > RandomPathAltChaser stype1=hungry stype2=powered cooldown=999 speed=0 img=oryx/ghost1 cons=4\n"
    )
    pac = pac.replace(marker, insert, 1)
    pac = replace_in_interaction(
        pac,
        "        hungry power > transformToAll stype=redOk stypeTo=redSc\n        hungry power > transformToAll stype=pinkOk stypeTo=pinkSc\n        hungry power > transformToAll stype=blueOk stypeTo=blueSc\n        hungry power > transformToAll stype=orangeOk stypeTo=orangeSc\n\n        hungry power > addTimer timer=200 ftype=transformToAll stype=redSc stypeTo=redOk killSecond=True\n        hungry power > addTimer timer=200 ftype=transformToAll stype=pinkSc stypeTo=pinkOk killSecond=True\n        hungry power > addTimer timer=200 ftype=transformToAll stype=blueSc stypeTo=blueOk killSecond=True\n        hungry power > addTimer timer=200 ftype=transformToAll stype=orangeSc stypeTo=orangeOk killSecond=True\n\n        hungry power > addTimer timer=200 ftype=transformToAll stype=powered stypeTo=hungry\n        hungry power > transformTo stype=powered",
        "        hungry power > transformToAll stype=redOk stypeTo=redFreeze\n        hungry power > transformToAll stype=pinkOk stypeTo=pinkFreeze\n        hungry power > transformToAll stype=blueOk stypeTo=blueFreeze\n        hungry power > transformToAll stype=orangeOk stypeTo=orangeFreeze\n\n        hungry power > addTimer timer=80 ftype=transformToAll stype=redFreeze stypeTo=redOk killSecond=True\n        hungry power > addTimer timer=80 ftype=transformToAll stype=pinkFreeze stypeTo=pinkOk killSecond=True\n        hungry power > addTimer timer=80 ftype=transformToAll stype=blueFreeze stypeTo=blueOk killSecond=True\n        hungry power > addTimer timer=80 ftype=transformToAll stype=orangeFreeze stypeTo=orangeOk killSecond=True\n\n        hungry power > addTimer timer=80 ftype=transformToAll stype=powered stypeTo=hungry\n        hungry power > transformTo stype=powered",
    )
    write_variant("pacman", "pacman_rules_ghost_freeze_on_powerup", pac)


def oil_slowdown_variant() -> None:
    road = base_text("roadfighter")
    road = road.replace(
        "        moving >\n            avatar  > MovingAvatar speed=0.5 color=YELLOW  img=newset/car_red healthPoints=10 limitHealthPoints=20\n            cars >",
        "        moving >\n            cars >",
        1,
    )
    road = road.replace(
        "            statics > \n                fuel > Missile orientation=DOWN speed=1 img=newset/fuel\n                tree > Missile orientation=DOWN speed=1 img=newset/tree2",
        "            statics > \n                fuel > Missile orientation=DOWN speed=1 img=newset/fuel\n                tree > Missile orientation=DOWN speed=1 img=newset/tree2\n                oil  > Missile orientation=DOWN speed=0.75 img=oryx/slime1\n            driver >\n                avatar  > MovingAvatar speed=0.5 color=YELLOW  img=newset/car_red healthPoints=10 limitHealthPoints=20\n                avatarSlow > MovingAvatar speed=0.25 color=YELLOW img=newset/car_red healthPoints=10 limitHealthPoints=20",
        1,
    )
    road = road.replace(
        "            slowPortal   > stype=carSlow cooldown=50  total=16\n            fastPortal   > stype=carFast cooldown=100 total=8\n            fuelPortal   > stype=fuel    cooldown=25  total=32\n            treePortal   > stype=tree    cooldown=2   total=400",
        "            slowPortal   > stype=carSlow cooldown=50  total=16\n            fastPortal   > stype=carFast cooldown=100 total=8\n            fuelPortal   > stype=fuel    cooldown=25  total=32\n            treePortal   > stype=tree    cooldown=2   total=400\n            oilPortal    > stype=oil     cooldown=35  total=24",
        1,
    )
    road = replace_in_interaction(
        road,
        "        avatar fuel > addHealthPoints value=10 killSecond=True\n        avatar EOS  > stepBack\n        avatar cars tree > killSprite",
        "        avatar fuel > addHealthPoints value=10 killSecond=True\n        avatar oil  > addTimer timer=24 ftype=transformToAll stype=avatarSlow stypeTo=avatar forceOrientation=True\n        avatar oil  > transformTo stype=avatarSlow killSecond=True forceOrientation=True\n        avatar EOS  > stepBack\n        avatar cars tree > killSprite",
    )
    road = replace_in_interaction(road, "        statics EOS > killSprite", "        statics EOS > killSprite\n        oil EOS > killSprite")
    road = road.replace("        f > fuelPortal floor\n", "        f > fuelPortal floor\n        o > oilPortal floor\n")
    write_variant("roadfighter", "roadfighter_rules_oil_slowdown", road)


def main() -> None:
    copy_legacy_games()
    quick_dash_variants()
    big_shot_variants()
    multishot_variants()
    pierce_variants()
    wall_on_death_variants()
    speed_variants()
    shield_reflect_variants()
    explosion_variants()
    pacman_freeze_variant()
    oil_slowdown_variant()
    write_catalog()
    write_marker()


if __name__ == "__main__":
    main()
