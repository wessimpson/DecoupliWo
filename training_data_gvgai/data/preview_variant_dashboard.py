"""Generate an animated HTML dashboard for games_world_model variants."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections import Counter
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer

import numpy as np
from PIL import Image

from training_data_gvgai.data.gvgai_jpype_env import GVGAIFileEnv
from training_data_gvgai.run_random_action import _ensure_build, _obs_rgb


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
GAMES_ROOT = PACKAGE_ROOT / "gvgai" / "gym_gvgai" / "envs" / "games_world_model"
GVGAI_JAVA_ROOT = PACKAGE_ROOT / "gvgai" / "gym_gvgai" / "envs" / "gvgai"
WORLD_MODEL_PATHS = PACKAGE_ROOT / "gvgai" / "gym_gvgai" / "envs" / "world_model_paths.py"


DESCRIPTIONS = {
    "default": "Base rules for comparison.",
    "quick_dash_2tile": "Avatar movement speed increased 2x.",
    "quick_dash_3tile": "Avatar movement speed increased 3x.",
    "quick_dash_5tile": "Avatar movement speed increased 5x.",
    "big_shot_1x": "Projectile shrink factor scaled to make a larger shot baseline.",
    "big_shot_2x": "Projectile size increased 2x.",
    "big_shot_4x": "Projectile size increased 4x.",
    "multishot_2": "Avatar fires two simultaneous cloned projectiles.",
    "multishot_3": "Avatar fires three simultaneous cloned projectiles.",
    "multishot_5": "Avatar fires five simultaneous cloned projectiles.",
    "pierce_shot": "Projectile defeats target without being consumed, where applicable.",
    "big_explosion_3rad": "Hits create a larger explosion effect around the target.",
    "big_explosion_5rad": "Missilecommand explosion effect increased further.",
    "enemy_speed_2x": "Enemy movement/cooldown tuned to make enemies faster.",
    "car_speed_2x": "Traffic/car hazards move faster.",
    "ghost_speed_2x": "Pac-Man ghosts move faster in normal and frightened states.",
    "ghost_freeze_on_powerup": "Power pellet freezes ghosts briefly instead of normal frightened movement.",
    "wall_on_death": "Killed enemies transform into walls/obstacles.",
    "shield_reflect": "Incoming enemy shot/contact is reflected or blocked instead of killing the avatar.",
    "oil_slowdown": "Roadfighter adds oil hazards that temporarily slow the player.",
    "enemy_explode": "Killed enemies spawn blast sprites around the hit.",
    "enemy_multishot": "Enemies fire multiple spread projectiles.",
    "multishot": "Legacy triple-shot player weapon.",
    "ricochet": "Player shot reverses after hitting an enemy.",
    "shoot_walls": "Player fires projectiles that create destroyable walls/obstacles.",
    "split_orthogonal": "Projectile hit spawns orthogonal side projectiles.",
    "two_hit_color": "Enemies need two hits and change color/state after the first hit.",
}

PROJECTILE_TAGS = (
    "big_shot",
    "multishot",
    "pierce_shot",
    "big_explosion",
    "wall_on_death",
    "enemy_explode",
    "ricochet",
    "shoot_walls",
    "split_orthogonal",
    "two_hit_color",
)
MOTION_TAGS = (
    "quick_dash",
    "enemy_speed",
    "car_speed",
    "ghost_speed",
    "ghost_freeze_on_powerup",
    "oil_slowdown",
    "shield_reflect",
    "enemy_multishot",
)


def load_world_model_paths():
    spec = importlib.util.spec_from_file_location("world_model_paths", WORLD_MODEL_PATHS)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {WORLD_MODEL_PATHS}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def level_indices(game_dir: Path) -> list[int]:
    return sorted(
        int(path.stem[3:])
        for path in game_dir.glob("lvl*.txt")
        if path.stem[3:].isdigit()
    )


def variant_stems(game_dir: Path) -> list[str]:
    base = game_dir.name
    return [base] + sorted(path.stem for path in game_dir.glob(f"{base}_rules_*.txt"))


def tag_for(base: str, stem: str) -> str:
    if stem == base:
        return "default"
    return stem.split("_rules_", 1)[1]


def selected_game_dirs(games: list[str]) -> list[Path]:
    dirs = sorted(path for path in GAMES_ROOT.iterdir() if path.is_dir())
    if not games:
        return dirs
    wanted = set(games)
    missing = sorted(wanted - {path.name for path in dirs})
    if missing:
        raise ValueError(f"Unknown game(s): {', '.join(missing)}")
    return [path for path in dirs if path.name in wanted]


def resize_frame(rgb: np.ndarray, max_side: int) -> Image.Image:
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    scale = max(1, max_side // max(img.size))
    return img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)


def save_gif(frames: list[np.ndarray], path: Path, max_side: int, frame_ms: int) -> None:
    images = [resize_frame(frame, max_side) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=frame_ms,
        loop=0,
        optimize=False,
    )


def _action_index(actions: list[str], action: str) -> int | None:
    try:
        return actions.index(action)
    except ValueError:
        return None


def _first_available(actions: list[str], candidates: list[str]) -> int:
    for action in candidates:
        idx = _action_index(actions, action)
        if idx is not None:
            return idx
    return 0


def _demo_action(actions: list[str], tag: str, tick: int) -> int | None:
    """Pick actions that make the named rule visible in short GIF rollouts."""
    if tag == "default":
        return None

    if tag.startswith(PROJECTILE_TAGS) or tag == "multishot":
        aim_fire_loop = [
            "ACTION_UP",
            "ACTION_USE",
            "ACTION_USE",
            "ACTION_RIGHT",
            "ACTION_USE",
            "ACTION_USE",
            "ACTION_DOWN",
            "ACTION_USE",
            "ACTION_USE",
            "ACTION_LEFT",
            "ACTION_USE",
            "ACTION_USE",
        ]
        action = aim_fire_loop[tick % len(aim_fire_loop)]
        idx = _action_index(actions, action)
        if idx is not None:
            return idx
        return _first_available(actions, ["ACTION_USE", "ACTION_UP", "ACTION_RIGHT", "ACTION_DOWN", "ACTION_LEFT"])

    if tag.startswith(MOTION_TAGS):
        movement_loop = [
            "ACTION_RIGHT",
            "ACTION_RIGHT",
            "ACTION_RIGHT",
            "ACTION_DOWN",
            "ACTION_DOWN",
            "ACTION_LEFT",
            "ACTION_LEFT",
            "ACTION_LEFT",
            "ACTION_UP",
            "ACTION_UP",
        ]
        action = movement_loop[tick % len(movement_loop)]
        return _first_available(actions, [action, "ACTION_RIGHT", "ACTION_DOWN", "ACTION_LEFT", "ACTION_UP"])

    return None


def run_variant(
    resolver,
    stem: str,
    level: int,
    *,
    steps: int,
    mcts_ms: int,
    policy: str,
) -> tuple[list[np.ndarray], str]:
    game_file, level_files = resolver.resolve_gvgai_paths(str(GAMES_ROOT.parent.resolve()), stem, 0)
    env = GVGAIFileEnv(
        game_file,
        level_files,
        level=level,
        gvgai_root=GVGAI_JAVA_ROOT,
        max_episode_steps=steps,
    )
    frames: list[np.ndarray] = []
    try:
        obs, info = env.reset()
        frames.append(_obs_rgb(obs))
        status = f"reset {info.get('winner', '')}"
        actions = env.get_action_meanings()
        tag = tag_for(stem.split("_rules_", 1)[0], stem)
        scripted_steps = 0
        mcts_steps = 0
        for tick in range(steps):
            action = _demo_action(actions, tag, tick) if policy == "demo" else None
            if action is None:
                obs, reward, terminated, truncated, info = env.step_mcts(mcts_ms)
                mcts_steps += 1
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                scripted_steps += 1
            frames.append(_obs_rgb(obs))
            status = f"t{info.get('game_tick', '?')} r={reward:.0f} {info.get('winner', '')}"
            if terminated or truncated:
                break
        return frames, f"{len(frames)} frames, {policy} policy ({scripted_steps} scripted/{mcts_steps} MCTS), {status}"
    finally:
        env.close()


def render_html(records: list[dict[str, object]], failures: list[dict[str, object]], steps: int, policy: str) -> str:
    records_json = json.dumps(records)
    failures_json = json.dumps(failures)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>GVGAI Variant Live Runs</title>
<style>
:root {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
body {{ margin: 0; background: #f6f7f8; color: #171a1f; }}
header {{ position: sticky; top: 0; z-index: 2; background: #fff; border-bottom: 1px solid #d8dde3; padding: 14px 18px; }}
h1 {{ font-size: 20px; margin: 0 0 8px; }}
.controls {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
select, input, button {{ font: inherit; padding: 7px 9px; border: 1px solid #b9c0c8; border-radius: 6px; background: white; }}
main {{ padding: 18px; }}
.summary {{ margin-bottom: 14px; color: #4d5967; font-size: 14px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }}
.card {{ background: #fff; border: 1px solid #d8dde3; border-radius: 8px; overflow: hidden; }}
.card img {{ display: block; width: 100%; height: 190px; object-fit: contain; background: #fbfbfc; image-rendering: pixelated; }}
.meta {{ padding: 10px 11px 12px; }}
.name {{ font-weight: 650; font-size: 14px; margin-bottom: 4px; word-break: break-word; }}
.desc {{ font-size: 13px; color: #394452; line-height: 1.35; min-height: 35px; }}
.status {{ margin-top: 8px; font-size: 12px; color: #697586; }}
.fail {{ margin: 12px 0; padding: 10px; background: #fff4f2; border: 1px solid #e2aaa3; border-radius: 8px; }}
</style>
</head>
<body>
<header>
<h1>GVGAI Variant Rule Demos - {steps}-step {policy} rollouts</h1>
<div class="controls">
<select id="game"></select>
<input id="query" placeholder="Filter variants" />
<button id="all">All games</button>
</div>
</header>
<main>
<div id="summary" class="summary"></div>
<div id="failures"></div>
<div id="grid" class="grid"></div>
</main>
<script>
const records = {records_json};
const failures = {failures_json};
const gameSelect = document.getElementById('game');
const query = document.getElementById('query');
const grid = document.getElementById('grid');
const summary = document.getElementById('summary');
const failBox = document.getElementById('failures');
const games = [...new Set(records.map(r => r.game))];
gameSelect.insertAdjacentHTML('afterbegin', '<option value="">All games</option>');
for (const g of games) {{
  const opt = document.createElement('option');
  opt.value = g;
  opt.textContent = g;
  gameSelect.appendChild(opt);
}}
function render() {{
  const q = query.value.toLowerCase().trim();
  const game = gameSelect.value;
  const shown = records.filter(r => (!game || r.game === game) && (!q || r.tag.toLowerCase().includes(q) || r.description.toLowerCase().includes(q)));
  summary.textContent = `${{records.length}} animated {steps}-step {policy} runs generated. Showing ${{shown.length}}. Failures: ${{failures.length}}.`;
  failBox.innerHTML = failures.length ? `<div class="fail">${{failures.map(f => `${{f.game}}/${{f.tag}}: ${{f.error}}`).join('<br>')}}</div>` : '';
  grid.innerHTML = '';
  for (const r of shown) {{
    const el = document.createElement('article');
    el.className = 'card';
    el.innerHTML = `<img src="${{r.gif}}?v={steps}" alt="${{r.game}} ${{r.tag}} animation" loading="lazy" />
      <div class="meta"><div class="name">${{r.game}} / ${{r.tag}}</div>
      <div class="desc">${{r.description}}</div>
      <div class="status">level ${{r.level}} - ${{r.status}}</div></div>`;
    grid.appendChild(el);
  }}
}}
document.getElementById('all').onclick = () => {{ gameSelect.value = ''; render(); }};
gameSelect.onchange = render;
query.oninput = render;
render();
</script>
</body>
</html>
"""


def serve_dashboard(output_dir: Path, port: int) -> None:
    os.chdir(output_dir)
    with TCPServer(("127.0.0.1", port), SimpleHTTPRequestHandler) as server:
        print(f"Serving variant dashboard at http://127.0.0.1:{port}/index.html")
        server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default="", help="Comma-separated games to render; default is all.")
    parser.add_argument("--steps", type=int, default=120, help="Environment steps per animated run.")
    parser.add_argument("--mcts-ms", type=int, default=5, help="MCTS budget per step in milliseconds.")
    parser.add_argument(
        "--policy",
        choices=("demo", "mcts"),
        default="demo",
        help="Action policy for GIFs. demo intentionally fires/moves to reveal rule variants; mcts uses only Java sampleMCTS.",
    )
    parser.add_argument("--level", type=int, default=None, help="Level index to preview; default is the first level present per game.")
    parser.add_argument("--output-dir", default="/tmp/decoupliwo_variant_live", help="Directory for index.html and GIF assets.")
    parser.add_argument("--max-side", type=int, default=192, help="Maximum source frame side before nearest-neighbor scaling.")
    parser.add_argument("--frame-ms", type=int, default=95, help="GIF frame duration in milliseconds.")
    parser.add_argument("--serve", action="store_true", help="Start a local HTTP server after generating the dashboard.")
    parser.add_argument("--port", type=int, default=8769, help="Local port for --serve.")
    parser.add_argument("--skip-build", action="store_true", help="Do not build GVGAI Java classes first.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_build:
        _ensure_build()

    output_dir = Path(args.output_dir).expanduser().resolve()
    asset_dir = output_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "variant_dashboard.log"
    resolver = load_world_model_paths()
    games = [part.strip() for part in args.games.split(",") if part.strip()]

    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    orig_out = os.dup(1)
    orig_err = os.dup(2)
    log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    try:
        for game_dir in selected_game_dirs(games):
            base = game_dir.name
            levels = level_indices(game_dir)
            if not levels:
                continue
            level = args.level if args.level is not None else levels[0]
            if level not in levels:
                failures.append({"game": base, "tag": "*", "level": level, "error": "level not present"})
                continue
            for stem in variant_stems(game_dir):
                tag = tag_for(base, stem)
                gif_name = f"{base}__{tag}.gif".replace("/", "_")
                try:
                    frames, status = run_variant(
                        resolver,
                        stem,
                        level,
                        steps=args.steps,
                        mcts_ms=args.mcts_ms,
                        policy=args.policy,
                    )
                    save_gif(frames, asset_dir / gif_name, args.max_side, args.frame_ms)
                    records.append(
                        {
                            "game": base,
                            "tag": tag,
                            "stem": stem,
                            "level": level,
                            "gif": f"assets/{gif_name}",
                            "status": status,
                            "description": DESCRIPTIONS.get(tag, "Rule variant present in games_world_model."),
                        }
                    )
                except Exception as exc:
                    failures.append({"game": base, "tag": tag, "level": level, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        os.dup2(orig_out, 1)
        os.dup2(orig_err, 2)
        os.close(log_fd)
        os.close(orig_out)
        os.close(orig_err)

    (output_dir / "index.html").write_text(render_html(records, failures, args.steps, args.policy), encoding="utf-8")
    summary = {
        "records": len(records),
        "steps": args.steps,
        "policy": args.policy,
        "failures": failures,
        "by_game": dict(sorted(Counter(record["game"] for record in records).items())),
        "html": str(output_dir / "index.html"),
        "log": str(log_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if failures:
        raise SystemExit(1)
    if args.serve:
        serve_dashboard(output_dir, args.port)


if __name__ == "__main__":
    main()
