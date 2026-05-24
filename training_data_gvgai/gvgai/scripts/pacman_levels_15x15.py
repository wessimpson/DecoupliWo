"""Hand-designed 15x15 pacman levels scaled from pacman_v0 originals."""

from __future__ import annotations

from collections import deque
from pathlib import Path

SIZE = 15
WALL = "w"

# Classic symmetric maze (original lvl0): corner power pellets, side fruit, central ghost pen,
# side tunnels on the ghost row, pacman spawn below the pen.
LVL0 = [
    "wwwwwwwwwwwwwww",
    "w0...........0w",
    "w.ww.wwfww.ww.w",
    "w.w...www...w.w",
    "w....+www+....w",
    "www.w+++++w.www",
    "w..+1234+.....w",
    "www.w+++++w.www",
    "w....+www+....w",
    "w.w...www...w.w",
    "w...ww.ww.ww..w",
    "w.w....A....w.w",
    "w...ww.ww.ww..w",
    "w0...........0w",
    "wwwwwwwwwwwwwww",
]

# Side chambers and narrow vertical lanes (original lvl1).
LVL1 = [
    "wwwwwwwwwwwwwww",
    "w0...........0w",
    "w.wwww...wwww.w",
    "w.w...w.w...w.w",
    "w.w.w.w.w.w.w.w",
    "w..ww.+++ww...w",
    "www.w+++++w.www",
    "w..+1234+.....w",
    "www.w+++++w.www",
    "w..ww.+++ww...w",
    "w.w.w.w.w.w.w.w",
    "w.w...w.w...w.w",
    "w.wwww.A.wwww.w",
    "w0....f......0w",
    "wwwwwwwwwwwwwww",
]

# Horizontal lanes with top/bottom power tunnels (original lvl2).
LVL2 = [
    "wwwwwwwwwwwwwww",
    "w0...........0w",
    "w.www.www.www.w",
    "w.............w",
    "w....+www+....w",
    "www.w+++++w.www",
    "w..+1234+.....w",
    "www.w+++++w.www",
    "w....+www+....w",
    "w.............w",
    "w.wwwfwww.www.w",
    "w.....A.......w",
    "w.............w",
    "w0...........0w",
    "wwwwwwwwwwwwwww",
]

# Large open west wing, compact east wing (original lvl3).
LVL3 = [
    "wwwwwwwwwwwwwww",
    "w.wwwwwwwwwww.w",
    "w0............w",
    "w.............w",
    "w.wwwwwwwwwww.w",
    "w.ww.+++++.ww.w",
    "w..+1234+.....w",
    "w.ww.+++++.ww.w",
    "w.............w",
    "w.............w",
    "w.wwwwwwwwwww.w",
    "w.............w",
    "w......A......w",
    "w0...........0w",
    "wwwwwwwwwwwwwww",
]

# Compact ghost box, wide lower playfield (original lvl4).
LVL4 = [
    "wwwwwwwwwwwwwww",
    "wwwwwwwwwwwwwww",
    "w0w.........w0w",
    "w.w.wwwwwww.w.w",
    "w.w.wwwwwww.w.w",
    "w....+www+....w",
    "www.w+++++w.www",
    "w..+1234+.....w",
    "www.w+++++w.www",
    "w....+www+....w",
    "w.............w",
    "w.wwwwwwwwwww.w",
    "w.....A.......w",
    "w0....f......0w",
    "wwwwwwwwwwwwwww",
]

LEVELS = [LVL0, LVL1, LVL2, LVL3, LVL4]


def _validate(rows: list[str], name: str) -> None:
    if len(rows) != SIZE or any(len(row) != SIZE for row in rows):
        bad = [(i, len(row)) for i, row in enumerate(rows) if len(row) != SIZE]
        raise ValueError(f"{name}: expected {SIZE}x{SIZE}, bad rows {bad}")

    if not all(rows[0][x] == WALL and rows[SIZE - 1][x] == WALL for x in range(SIZE)):
        raise ValueError(f"{name}: missing top/bottom border")
    if not all(rows[y][0] == WALL and rows[y][SIZE - 1] == WALL for y in range(SIZE)):
        raise ValueError(f"{name}: missing side border")

    if sum(row.count("A") for row in rows) != 1:
        raise ValueError(f"{name}: expected exactly one pacman spawn")
    for ghost in "1234":
        if sum(row.count(ghost) for row in rows) != 1:
            raise ValueError(f"{name}: expected exactly one '{ghost}' spawn")

    start = next((y, x) for y, row in enumerate(rows) for x, ch in enumerate(row) if ch == "A")
    seen: set[tuple[int, int]] = {start}
    queue = deque([start])
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < SIZE and 0 <= nx < SIZE and rows[ny][nx] != WALL and (ny, nx) not in seen:
                seen.add((ny, nx))
                queue.append((ny, nx))

    food = {(y, x) for y, row in enumerate(rows) for x, ch in enumerate(row) if ch in ".0f"}
    if not food.issubset(seen):
        unreachable = len(food - seen)
        raise ValueError(f"{name}: {unreachable} food cells unreachable from pacman")


def write_pacman_levels(root: Path | None = None) -> None:
    root = root or Path(__file__).resolve().parents[1] / "gym_gvgai" / "envs" / "games_world_model" / "pacman"
    for index, rows in enumerate(LEVELS):
        name = f"lvl{index}.txt"
        _validate(rows, name)
        (root / name).write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_pacman_levels()
    print("Wrote 5 pacman 15x15 levels")
