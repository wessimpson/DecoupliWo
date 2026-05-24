from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GAMES_ROOT = ROOT / "training_data_gvgai" / "gvgai" / "gym_gvgai" / "envs" / "games"
WM_ROOT = ROOT / "training_data_gvgai" / "gvgai" / "gym_gvgai" / "envs" / "games_world_model"
CATALOG_PATH = ROOT / "training_data_gvgai" / "data" / "gvgai_variant_catalog.world_model.json"
MARKER_PATH = WM_ROOT / ".built_from_data_collection"
MARKER_VERSION = "v15_world_model_variants_15x15"
LEVEL_SIZE = 15
LEVELS = tuple(range(5))

PACMAN_WALL = "w"
PACMAN_GHOST = set("1234")
PACMAN_CRITICAL = set("A01234f")


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

CRITICAL_LEVEL_CHARS = {
    "frogs": set("ABg1234-xl_="),
    "ikaruga": set("Aqwerzx"),
    "missilecommand": set("Acmf"),
    "pacman": set("A01234f"),
    "roadfighter": set("Aftsctx"),
    "seaquest": set("A1234"),
    "sheriff": set("A01234udlrs"),
}

UNIQUE_LEVEL_CHARS = {
    "frogs": set("AB"),
    "ikaruga": set("A"),
    "missilecommand": set("A"),
    "pacman": set("A"),
    "roadfighter": set("A"),
    "seaquest": set("A"),
    "sheriff": set("A"),
}

LEVEL_CHAR_PRIORITY = {
    ".": 0,
    "+": 1,
    "a": 1,
    "w": 3,
    "W": 3,
    "o": 3,
    "0": 20,
    "1": 30,
    "2": 30,
    "3": 30,
    "4": 30,
    "A": 100,
    "B": 100,
    "g": 80,
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


def rescale_missilecommand_incoming_speeds(text: str) -> str:
    """square_size=8 truncates 0.1/0.3 to zero grid steps; use 1.0/3.0 (same 1:3 ratio)."""
    text = text.replace("incoming_slow  > Chaser stype=city color=ORANGE speed=0.1", "incoming_slow  > Chaser stype=city color=ORANGE speed=1.0")
    text = text.replace("incoming_fast  > Chaser stype=city color=YELLOW speed=0.3", "incoming_fast  > Chaser stype=city color=YELLOW speed=3.0")
    return text


def normalize_legacy_base_text(game: str, text: str) -> str:
    text = ensure_square_size_8(text)
    if game == "missilecommand":
        text = rescale_missilecommand_incoming_speeds(text)
        if not re.search(r"(?m)^\s*w\s*>", text):
            text = text.replace("    A > floor avatar", "    A > floor avatar\n    w > floor wall", 1)
    return text


def normalize_level_mapping_value(value: str) -> str:
    return " ".join(value.split("#", 1)[0].split())


def parse_level_mapping(text: str) -> dict[str, str]:
    start, end = section_bounds(text, "LevelMapping")
    mapping: dict[str, str] = {}
    for line in text[start:end].splitlines():
        match = re.match(r"\s*(\S)\s*>\s*(.*)$", line)
        if match:
            mapping[match.group(1)] = normalize_level_mapping_value(match.group(2))
    return mapping


def level_char_map(source_game: str, target_game: str) -> dict[str, str]:
    source_map = parse_level_mapping(read_text(GAMES_ROOT / f"{source_game}_v0" / f"{source_game}.txt"))
    target_map = parse_level_mapping(read_text(WM_ROOT / target_game / f"{target_game}.txt"))
    target_by_value = {value: char for char, value in target_map.items()}
    return {
        source_char: target_by_value.get(value, source_char if source_char in target_map else source_char)
        for source_char, value in source_map.items()
    }


def normalize_level_rows(text: str, char_map: dict[str, str]) -> list[list[str]]:
    rows = [
        [char_map.get(ch, ch) for ch in line]
        for line in text.replace("\r", "").splitlines()
        if line
    ]
    if not rows:
        raise ValueError("Cannot resize an empty level")
    width = max(len(row) for row in rows)
    fill = max((ch for row in rows for ch in row), key=lambda ch: sum(r.count(ch) for r in rows))
    return [row + [fill] * (width - len(row)) for row in rows]


def choose_level_char(cells: list[str], critical: set[str]) -> str:
    counts = {ch: cells.count(ch) for ch in set(cells)}
    return max(
        counts,
        key=lambda ch: (
            counts[ch],
            LEVEL_CHAR_PRIORITY.get(ch, 40 if ch in critical else 10),
        ),
    )


def place_preserved_char(grid: list[list[str]], char: str, target_y: int, target_x: int, critical: set[str]) -> None:
    size = len(grid)
    if grid[target_y][target_x] not in critical or grid[target_y][target_x] == char:
        grid[target_y][target_x] = char
        return

    for radius in range(1, size):
        for y in range(max(0, target_y - radius), min(size, target_y + radius + 1)):
            for x in range(max(0, target_x - radius), min(size, target_x + radius + 1)):
                if abs(y - target_y) + abs(x - target_x) > radius:
                    continue
                if grid[y][x] not in critical:
                    grid[y][x] = char
                    return


def resize_level_to_15(game: str, text: str, char_map: dict[str, str]) -> str:
    rows = normalize_level_rows(text, char_map)
    height = len(rows)
    width = len(rows[0])
    critical = CRITICAL_LEVEL_CHARS.get(game, set())
    unique = UNIQUE_LEVEL_CHARS.get(game, set())
    grid: list[list[str]] = []

    for target_y in range(LEVEL_SIZE):
        source_y0 = target_y * height // LEVEL_SIZE
        source_y1 = max(source_y0 + 1, ((target_y + 1) * height + LEVEL_SIZE - 1) // LEVEL_SIZE)
        row: list[str] = []
        for target_x in range(LEVEL_SIZE):
            source_x0 = target_x * width // LEVEL_SIZE
            source_x1 = max(source_x0 + 1, ((target_x + 1) * width + LEVEL_SIZE - 1) // LEVEL_SIZE)
            cells = [
                rows[y][x]
                for y in range(source_y0, min(source_y1, height))
                for x in range(source_x0, min(source_x1, width))
            ]
            row.append(choose_level_char(cells, critical))
        grid.append(row)

    source_unique_counts = {
        char: sum(row.count(char) for row in rows)
        for char in unique
    }
    replacement = choose_level_char([ch for row in grid for ch in row if ch not in unique] or ["."], critical)
    for y, row in enumerate(grid):
        for x, char in enumerate(row):
            if char in unique and source_unique_counts.get(char) == 1:
                grid[y][x] = replacement

    for source_y, row in enumerate(rows):
        for source_x, char in enumerate(row):
            if char not in critical:
                continue
            target_y = min(LEVEL_SIZE - 1, int((source_y + 0.5) * LEVEL_SIZE / height))
            target_x = min(LEVEL_SIZE - 1, int((source_x + 0.5) * LEVEL_SIZE / width))
            place_preserved_char(grid, char, target_y, target_x, critical)

    return "\n".join("".join(row) for row in grid)


def _level_block(rows: list[list[str]], ty: int, tx: int, height: int, width: int) -> list[str]:
    source_y0 = ty * height // LEVEL_SIZE
    source_y1 = max(source_y0 + 1, ((ty + 1) * height + LEVEL_SIZE - 1) // LEVEL_SIZE)
    source_x0 = tx * width // LEVEL_SIZE
    source_x1 = max(source_x0 + 1, ((tx + 1) * width + LEVEL_SIZE - 1) // LEVEL_SIZE)
    return [
        rows[y][x]
        for y in range(source_y0, min(source_y1, height))
        for x in range(source_x0, min(source_x1, width))
    ]


def _level_center_index(ty: int, tx: int, height: int, width: int) -> tuple[int, int]:
    return (
        min(height - 1, int((ty + 0.5) * height / LEVEL_SIZE)),
        min(width - 1, int((tx + 0.5) * width / LEVEL_SIZE)),
    )


def _nearest_open_cell(
    grid: list[list[str]],
    target_y: int,
    target_x: int,
    occupied: set[tuple[int, int]],
) -> tuple[int, int]:
    size = len(grid)
    if grid[target_y][target_x] != PACMAN_WALL and (target_y, target_x) not in occupied:
        return target_y, target_x
    for radius in range(1, size):
        for y in range(max(0, target_y - radius), min(size, target_y + radius + 1)):
            for x in range(max(0, target_x - radius), min(size, target_x + radius + 1)):
                if abs(y - target_y) + abs(x - target_x) > radius:
                    continue
                if grid[y][x] != PACMAN_WALL and (y, x) not in occupied:
                    return y, x
    return target_y, target_x


def _place_pacman_ghost_spawns(grid: list[list[str]]) -> None:
    plus_cells = [
        (y, x)
        for y in range(LEVEL_SIZE)
        for x in range(LEVEL_SIZE)
        if grid[y][x] == "+"
    ]
    if len(plus_cells) < 4:
        return
    rows = sorted({y for y, _ in plus_cells})
    mid_row = rows[len(rows) // 2]
    slots = sorted(
        [(y, x) for y, x in plus_cells if y == mid_row],
        key=lambda cell: cell[1],
    )
    if len(slots) < 4:
        slots = sorted(plus_cells, key=lambda cell: (cell[0], cell[1]))
    start = max(0, (len(slots) - 4) // 2)
    for index, char in enumerate("1234"):
        y, x = slots[start + index]
        grid[y][x] = char


def resize_pacman_level_to_15(text: str, char_map: dict[str, str]) -> str:
    """Downsample pacman mazes to 15x15 while keeping corridors and the ghost box."""
    rows = normalize_level_rows(text, char_map)
    height = len(rows)
    width = len(rows[0])
    grid: list[list[str]] = [["." for _ in range(LEVEL_SIZE)] for _ in range(LEVEL_SIZE)]
    ghost_zone = [[False] * LEVEL_SIZE for _ in range(LEVEL_SIZE)]

    for target_y in range(LEVEL_SIZE):
        for target_x in range(LEVEL_SIZE):
            cells = _level_block(rows, target_y, target_x, height, width)
            center_y, center_x = _level_center_index(target_y, target_x, height, width)
            center = rows[center_y][center_x]
            wall_fraction = cells.count(PACMAN_WALL) / len(cells)
            if wall_fraction >= 0.62 or (wall_fraction >= 0.45 and center == PACMAN_WALL):
                grid[target_y][target_x] = PACMAN_WALL
            elif "+" in cells or any(char in PACMAN_GHOST for char in cells):
                grid[target_y][target_x] = "+"
                ghost_zone[target_y][target_x] = True
            else:
                grid[target_y][target_x] = "."

    for index in range(LEVEL_SIZE):
        grid[0][index] = grid[LEVEL_SIZE - 1][index] = PACMAN_WALL
        grid[index][0] = grid[index][LEVEL_SIZE - 1] = PACMAN_WALL

    _place_pacman_ghost_spawns(grid)

    occupied: set[tuple[int, int]] = set()
    for source_y, row in enumerate(rows):
        for source_x, char in enumerate(row):
            if char not in PACMAN_CRITICAL or char in PACMAN_GHOST:
                continue
            target_y = min(LEVEL_SIZE - 1, int((source_y + 0.5) * LEVEL_SIZE / height))
            target_x = min(LEVEL_SIZE - 1, int((source_x + 0.5) * LEVEL_SIZE / width))
            cell_y, cell_x = _nearest_open_cell(grid, target_y, target_x, occupied)
            if grid[cell_y][cell_x] == "+":
                grid[cell_y][cell_x] = "."
            grid[cell_y][cell_x] = char
            occupied.add((cell_y, cell_x))

    special = PACMAN_CRITICAL | {"+"}
    for target_y in range(LEVEL_SIZE):
        for target_x in range(LEVEL_SIZE):
            if grid[target_y][target_x] == PACMAN_WALL:
                continue
            if grid[target_y][target_x] in special:
                continue
            grid[target_y][target_x] = "+" if ghost_zone[target_y][target_x] else "."

    return "\n".join("".join(row) for row in grid)


def copy_legacy_games() -> None:
    import importlib.util

    pacman_levels_path = (
        Path(__file__).resolve().parent.parent / "gvgai" / "scripts" / "pacman_levels_15x15.py"
    )
    pacman_spec = importlib.util.spec_from_file_location("pacman_levels_15x15", pacman_levels_path)
    pacman_levels = importlib.util.module_from_spec(pacman_spec)
    assert pacman_spec.loader is not None
    pacman_spec.loader.exec_module(pacman_levels)

    for game in LEGACY_ONLY:
        src_dir = GAMES_ROOT / f"{game}_v0"
        dst_dir = WM_ROOT / game
        dst_dir.mkdir(parents=True, exist_ok=True)
        base = normalize_legacy_base_text(game, read_text(src_dir / f"{game}.txt"))
        if game == "pacman":
            base = apply_pacman_power_rules(base)
        write_text(dst_dir / f"{game}.txt", ensure_square_size_8(base))
        char_map = level_char_map(game, game)
        if game == "pacman":
            pacman_levels.write_pacman_levels(dst_dir)
        else:
            for i in LEVELS:
                level = resize_level_to_15(game, read_text(src_dir / f"{game}_lvl{i}.txt"), char_map)
                write_text(dst_dir / f"lvl{i}.txt", level)
        for stale_level in dst_dir.glob("lvl[5-9]*.txt"):
            stale_level.unlink()


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


def sheriff_explosion_variant(base: str, radius: int) -> str:
    """Enemy death blast like aliens enemy_explode_* (fire sprite inside SpriteSet)."""
    explosion_line = (
        f"        explosion > Flicker limit=6 singleton=True img=oryx/fire1 shrinkfactor={radius}"
    )
    wall_line = "        wall > Immovable autotiling=True img=oryx/dirtwall"
    if re.search(r"(?m)^\s*explosion\s*>", base):
        text = re.sub(r"(?m)^\s*explosion\s*>.*$", explosion_line, base, count=1)
    else:
        text = base.replace(wall_line, explosion_line + "\n" + wall_line, 1)
    blast_rules = (
        "        bandit bullet > transformTo stype=explosion killSecond=True scoreChange=1\n"
        "        bandit explosion > killSprite\n"
        "        missile explosion > killSprite\n"
        "        bullet explosion > killSprite\n"
        "        avatar explosion > killSprite scoreChange=-1"
    )
    return replace_in_interaction(
        text,
        "        bandit bullet > killBoth scoreChange=1",
        blast_rules,
    )


PACMAN_POWER_TIMER = 120

_PACMAN_GHOST_OK_SCARED = (
    ("redOk", "redSc"),
    ("pinkOk", "pinkSc"),
    ("blueOk", "blueSc"),
    ("orangeOk", "orangeSc"),
)
_PACMAN_GHOST_OK_FREEZE = (
    ("redOk", "redFreeze"),
    ("pinkOk", "pinkFreeze"),
    ("blueOk", "blueFreeze"),
    ("orangeOk", "orangeFreeze"),
)


def _pacman_power_mode_lines(timer: int, ok_scared: tuple[tuple[str, str], ...]) -> list[str]:
    """Hungry-only power entry (matches pacman_v0); no powered-power re-scare (avoids dupes)."""
    lines: list[str] = []
    for ok, scared in ok_scared:
        lines.append(f"        hungry power > transformToAll stype={ok} stypeTo={scared}")
    for ok, scared in ok_scared:
        lines.append(
            f"        hungry power > addTimer timer={timer} ftype=transformToAll stype={scared} stypeTo={ok} killSecond=True"
        )
    lines.append(f"        hungry power > addTimer timer={timer} ftype=transformToAll stype=powered stypeTo=hungry")
    lines.append("        hungry power > transformTo stype=powered")
    return lines


def fix_pacman_ghost_singleton(text: str) -> str:
    """Singleton on *Ok only so redSc->redOk transform works while a scared ghost exists."""
    for color in ("red", "blue", "pink", "orange"):
        text = re.sub(rf"(?m)^(\s*{color}\s*>)\s*singleton=True\s*$", r"\1", text, count=1)
    def _ok_singleton(line: re.Match[str]) -> str:
        body = line.group(0).rstrip()
        return body if "singleton=True" in body else body + " singleton=True"

    return re.sub(r"(?m)^\s*(?:red|blue|pink|orange)Ok\s*>.*$", _ok_singleton, text)


def _strip_powered_power_mode_lines(text: str) -> str:
    text = re.sub(r"(?m)^[ \t]*powered power > transformToAll.*\n", "", text)
    return re.sub(r"(?m)^[ \t]*powered power > addTimer.*\n", "", text)


def _pacman_ghost_eaten_lines(ok_scared: tuple[tuple[str, str], ...]) -> list[str]:
    return [f"        {scared} powered > transformTo stype={ok} scoreChange=40" for ok, scared in ok_scared]


def apply_pacman_power_rules(text: str, timer: int = PACMAN_POWER_TIMER, use_freeze: bool = False) -> str:
    """Finite power-up; eaten ghosts become chasers (transformTo), not removed."""
    ok_scared = _PACMAN_GHOST_OK_FREEZE if use_freeze else _PACMAN_GHOST_OK_SCARED
    power_block = "\n".join(_pacman_power_mode_lines(timer, ok_scared))
    eat_block = "\n".join(_pacman_ghost_eaten_lines(ok_scared))
    text = _strip_powered_power_mode_lines(text)
    power_pattern = re.compile(
        r"(?ms)^[ \t]*hungry power > transformToAll.*?^[ \t]*hungry power > transformTo stype=powered[ \t]*$"
    )
    if power_pattern.search(text):
        text = power_pattern.sub(power_block, text, count=1)
    text = re.sub(
        r"(?m)^[ \t]*ghost powered > .*$",
        eat_block,
        text,
        count=1,
    )
    for ok, scared in ok_scared:
        text = text.replace(
            f"        {scared} powered > killSprite scoreChange=40",
            f"        {scared} powered > transformTo stype={ok} scoreChange=40",
        )
    return fix_pacman_ghost_singleton(text)


def sheriff_multishot_variant(base: str, count: int) -> str:
    """Fan centered on avatar facing (spreadDegrees); fixed UP orientations miss left/right/down."""
    if count not in (2, 3, 5):
        raise ValueError(f"unsupported sheriff multishot count: {count}")
    spread = "22.5" if count == 5 else "45"
    stype_field = ",".join(["bullet"] * count)
    text = re.sub(
        r"(?m)^\s*bullet\s*>.*$",
        "        bullet > Missile img=oryx/orb3 shrinkfactor=0.5 speed=1.0",
        base,
        count=1,
    )
    avatar_line = (
        f"        avatar  > ShootAvatar stype={stype_field} img=newset/sheriff1 "
        f"speed=1.0 alignShotToOrientation=True fireAllWeapons=True spreadPixels=0 "
        f"spreadDegrees={spread} rotateInPlace=False"
    )
    return re.sub(r"(?m)^\s*avatar\s*>.*$", avatar_line, text, count=1)


def add_sprite_before_levels(text: str, line: str, sprite_name: str) -> str:
    if re.search(rf"(?m)^\s*{re.escape(sprite_name)}\s*>", text):
        return text
    match = re.search(r"(?m)^\s*LevelMapping\s*$", text)
    if not match:
        raise ValueError("Missing LevelMapping section")
    sprite_line = line.rstrip()
    if not sprite_line.startswith("        "):
        sprite_line = "        " + sprite_line.lstrip()
    return text[: match.start()] + sprite_line + "\n\n" + text[match.start() :]


def clone_projectile_variant(text: str, avatar_sprite: str, projectile_sprite: str, count: int) -> str:
    names = [f"{projectile_sprite}{i}" for i in range(1, count + 1)]
    avatar_pattern = re.compile(rf"(?m)^(?P<line>\s*{re.escape(avatar_sprite)}\s*>\s*.*)$")
    avatar_match = avatar_pattern.search(text)
    if not avatar_match:
        raise ValueError(f"Missing avatar line for {avatar_sprite}")
    avatar_line = avatar_match.group("line")
    avatar_line = re.sub(rf"stype={re.escape(projectile_sprite)}\b", f"stype={','.join(names)}", avatar_line)
    if "fireAllWeapons=True" not in avatar_line:
        avatar_line += " fireAllWeapons=True"
    if "spreadPixels=" not in avatar_line and "spreadDegrees=" not in avatar_line:
        if projectile_sprite == "whiteBullet":
            avatar_line += " spreadPixels=8"
        elif projectile_sprite == "sam" and count == 3:
            avatar_line += " spreadPixels=4"
        else:
            avatar_line += " spreadPixels=0"
    if projectile_sprite == "sam" and count != 3 and "spreadDegrees=" not in avatar_line:
        avatar_line += " spreadDegrees=22.5" if count == 5 else " spreadDegrees=45"
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
        "defaults": {"levels": list(LEVELS)},
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
        "levels=lvl0..lvl4",
        "grid=15x15",
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
            text = scale_shrinkfactor(base, projectile, mult)
            # Ikaruga: black ship uses blackBullet (base 0.5); scale both polarities together.
            if game == "ikaruga":
                text = scale_shrinkfactor(text, "blackBullet", mult)
            write_variant(game, f"{game}_rules_big_shot_{mult}x", text)


def ikaruga_multishot_variant(base: str, count: int) -> str:
    """FlakAvatar multishot: explicit up-fan orientations, both polarities, no singleton on fan bullets."""
    if count == 2:
        white_names = ["whiteBulletL", "whiteBulletR"]
        black_names = ["blackBulletL", "blackBulletR"]
        white_lines = (
            "            whiteBulletL > orientation=-0.7071,-0.7071 color=BLUE img=oryx/cspell1\n"
            "            whiteBulletR > orientation=0.7071,-0.7071 color=BLUE img=oryx/cspell1\n"
        )
        black_lines = (
            "            blackBulletL > orientation=-0.7071,-0.7071 color=BLUE img=oryx/orb3 shrinkfactor=0.5\n"
            "            blackBulletR > orientation=0.7071,-0.7071 color=BLUE img=oryx/orb3 shrinkfactor=0.5\n"
        )
    elif count == 3:
        white_names = ["whiteBulletL", "whiteBulletC", "whiteBulletR"]
        black_names = ["blackBulletL", "blackBulletC", "blackBulletR"]
        white_lines = (
            "            whiteBulletL > orientation=-0.7071,-0.7071 color=BLUE img=oryx/cspell1\n"
            "            whiteBulletC > orientation=UP    color=BLUE img=oryx/cspell1\n"
            "            whiteBulletR > orientation=0.7071,-0.7071 color=BLUE img=oryx/cspell1\n"
        )
        black_lines = (
            "            blackBulletL > orientation=-0.7071,-0.7071 color=BLUE img=oryx/orb3 shrinkfactor=0.5\n"
            "            blackBulletC > orientation=UP    color=BLUE img=oryx/orb3 shrinkfactor=0.5\n"
            "            blackBulletR > orientation=0.7071,-0.7071 color=BLUE img=oryx/orb3 shrinkfactor=0.5\n"
        )
    elif count == 5:
        white_names = [f"whiteBullet{i}" for i in range(1, 6)]
        black_names = [f"blackBullet{i}" for i in range(1, 6)]
        orientations = [
            "-0.7071,-0.7071",
            "-0.3827,-0.9239",
            "UP",
            "0.3827,-0.9239",
            "0.7071,-0.7071",
        ]
        white_lines = "".join(
            f"            {n} > orientation={o} color=BLUE img=oryx/cspell1\n"
            for n, o in zip(white_names, orientations)
        )
        black_lines = "".join(
            f"            {n} > orientation={o} color=BLUE img=oryx/orb3 shrinkfactor=0.5\n"
            for n, o in zip(black_names, orientations)
        )
    else:
        raise ValueError(f"unsupported ikaruga multishot count: {count}")

    text = base
    text = re.sub(
        r"(?m)^(\s*)whiteAvatar\s*>.*$",
        f"\\1whiteAvatar > stype={','.join(white_names)} fireAllWeapons=True spreadPixels=0 img=oryx/spaceship1",
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^(\s*)blackAvatar\s*>.*$",
        f"\\1blackAvatar > stype={','.join(black_names)} fireAllWeapons=True spreadPixels=0 img=oryx/spaceship2",
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^\s*whiteBullet\s*>.*\n\s*blackBullet\s*>.*\n",
        white_lines + black_lines,
        text,
        count=1,
    )
    white_list = " ".join(white_names)
    black_list = " ".join(black_names)
    start, end = section_bounds(text, "InteractionSet")
    block = text[start:end]
    block = block.replace("bomb        whiteBullet > killBoth", f"bomb        {white_list} {black_list} > killBoth")
    block = block.replace(
        "blackAlien  blackBullet > killBoth scoreChange=1",
        f"blackAlien  {black_list} > killBoth scoreChange=1",
    )
    block = block.replace(
        "whiteAlien  whiteBullet > killBoth scoreChange=1",
        f"whiteAlien  {white_list} > killBoth scoreChange=1",
    )
    return text[:start] + block + text[end:]


def multishot_variants() -> None:
    for game in ("defender", "ikaruga", "seaquest", "missilecommand", "sheriff"):
        base = base_text(game)
        for count in (2, 3, 5):
            if game == "ikaruga":
                text = ikaruga_multishot_variant(base, count)
            elif game == "sheriff":
                text = sheriff_multishot_variant(base, count)
            else:
                avatar, projectile = PROJECTILES[game]
                text = clone_projectile_variant(base, avatar, projectile, count)
            write_variant(game, f"{game}_rules_multishot_{count}", text)


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

    ikaruga = add_sprite_before_levels(
        base_text("ikaruga"),
        "explosion > Flicker limit=6 singleton=True img=oryx/fire1 shrinkfactor=3",
        "explosion",
    )
    ikaruga = replace_in_interaction(
        ikaruga,
        "        blackAlien  blackBullet > killBoth scoreChange=1\n        whiteAlien  whiteBullet > killBoth scoreChange=1",
        "        blackAlien  blackBullet > transformTo stype=explosion killSecond=True scoreChange=1\n        whiteAlien  whiteBullet > transformTo stype=explosion killSecond=True scoreChange=1\n        alien       explosion   > killSprite\n        missile     explosion   > killSprite",
    )
    ikaruga = replace_in_interaction(
        ikaruga,
        "        avatar      bomb        > killBoth scoreChange=-1",
        "        avatar      bomb        > killBoth scoreChange=-1\n        avatar      explosion   > killSprite scoreChange=-1",
    )
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
    pac = apply_pacman_power_rules(pac, timer=80, use_freeze=True)
    write_variant("pacman", "pacman_rules_ghost_freeze_on_powerup", pac)


def refresh_pacman_variants() -> None:
    """Re-apply power rules to base and all pacman rule variants (except intentional wall_on_death eat)."""
    base = apply_pacman_power_rules(read_text(WM_ROOT / "pacman" / "pacman.txt"))
    write_text(WM_ROOT / "pacman" / "pacman.txt", ensure_square_size_8(base))
    view_path = WM_ROOT / "pacman" / "pacman_view.txt"
    if view_path.is_file():
        write_text(view_path, ensure_square_size_8(apply_pacman_power_rules(read_text(view_path))))
    for path in sorted((WM_ROOT / "pacman").glob("pacman_rules_*.txt")):
        if "wall_on_death" in path.name or "ghost_freeze" in path.name:
            continue
        write_text(path, ensure_square_size_8(apply_pacman_power_rules(read_text(path))))
    pacman_freeze_variant()


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
    refresh_pacman_variants()
    oil_slowdown_variant()
    write_catalog()
    write_marker()


if __name__ == "__main__":
    main()
