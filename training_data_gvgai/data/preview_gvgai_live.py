from __future__ import annotations

import argparse
import io
import json
import random
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from training_data_gvgai.data.collect_gvgai_jpype import (
	DEFAULT_CATALOG,
	choose_action,
	load_jobs,
	make_env,
	parse_filter,
)
from training_data_gvgai.data.frame_metadata import (
	FrameLabeler,
	action_semantic,
	build_frame_metadata,
	prepare_frame_inputs,
)


@dataclass
class LiveState:
	lock: threading.Lock = field(default_factory=threading.Lock)
	frame_png: bytes = b""
	metadata: dict[str, Any] = field(default_factory=dict)
	status: str = "starting"
	error: str | None = None
	frame_count: int = 0

	def snapshot(self) -> dict[str, Any]:
		with self.lock:
			return {
				"status": self.status,
				"error": self.error,
				"frame_count": self.frame_count,
				"ascii": self.metadata.get("frame", {}).get("ascii", {}),
				"metadata": clean_preview_metadata(self.metadata),
			}

	def image(self) -> bytes:
		with self.lock:
			return self.frame_png

	def update(self, frame_png: bytes, metadata: dict[str, Any]) -> None:
		with self.lock:
			self.frame_png = frame_png
			self.metadata = metadata
			self.frame_count += 1
			self.status = "running"
			self.error = None

	def fail(self, error: BaseException) -> None:
		with self.lock:
			self.status = "error"
			self.error = f"{type(error).__name__}: {error}"


def clean_preview_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
	if not metadata:
		return {}

	objects = metadata.get("frame", {}).get("objects", {})
	object_counts = {
		role: len(rows)
		for role, rows in sorted(objects.items())
		if rows
	}
	event_labels = sorted({
		label
		for event in metadata.get("events", [])
		for label in event.get("labels", [])
	})
	label_block = metadata.get("labels", {})
	player = label_block.get("player", {})
	enemy = label_block.get("enemy", {})
	labels = {
		"player_action": player.get("making"),
	}
	if player.get("receiving"):
		labels["player_receiving"] = player["receiving"]
	if enemy.get("receiving"):
		labels["enemy_receiving"] = enemy["receiving"]
	if event_labels:
		labels["events"] = event_labels

	cleaned = {
		"game": metadata.get("game"),
		"level": metadata.get("level"),
		"variant": metadata.get("rollout_variant"),
		"episode": {
			"id": metadata.get("episode_id"),
			"step": metadata.get("step_in_episode"),
			"seed": metadata.get("seed"),
		},
		"tick": {
			"current": metadata.get("tick"),
			"next": metadata.get("next_tick"),
		},
		"action": metadata.get("action", {}),
		"variants": {
			"configured": metadata.get("configured_variants", []),
			"active": metadata.get("active_variants", []),
			"active_flags": metadata.get("rule_flags_active", {}),
		},
		"labels": labels,
		"score": metadata.get("frame", {}).get("score"),
		"winner": metadata.get("frame", {}).get("winner"),
	}
	if object_counts:
		cleaned["object_counts"] = object_counts
	if metadata.get("events"):
		cleaned["events"] = metadata["events"]
	return cleaned


def as_png(obs: Any) -> bytes:
	arr = np.asarray(obs)
	if arr.ndim != 3:
		raise ValueError(f"expected HWC observation, got shape={arr.shape}")
	if arr.shape[-1] == 4:
		arr = arr[:, :, :3]
	if arr.dtype != np.uint8:
		arr = arr.clip(0, 255).astype(np.uint8)
	image = Image.fromarray(arr, mode="RGB")
	buf = io.BytesIO()
	image.save(buf, format="PNG")
	return buf.getvalue()


def html_page() -> bytes:
	return b"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>GVGAI Live Metadata Preview</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }
    body { margin: 0; background: #101214; color: #eef1f3; }
    header { display: flex; align-items: center; justify-content: space-between; padding: 14px 18px; border-bottom: 1px solid #2b3035; background: #171a1d; }
    h1 { margin: 0; font-size: 16px; font-weight: 650; }
    #status { color: #9fb0bd; font-size: 13px; }
    main { display: grid; grid-template-columns: minmax(360px, 42vw) 1fr; height: calc(100vh - 49px); }
    .stage { display: grid; place-items: center; padding: 18px; border-right: 1px solid #2b3035; background: #090a0b; }
    img { width: min(100%, 720px); image-rendering: pixelated; border: 1px solid #30363d; background: #000; }
    .meta { display: grid; grid-template-rows: auto auto 1fr; min-width: 0; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; padding: 14px 18px; border-bottom: 1px solid #2b3035; }
    .chip { padding: 5px 8px; border: 1px solid #3b444d; border-radius: 6px; background: #1a1f24; font-size: 12px; color: #d8e0e6; }
    .chip.active { border-color: #3fa777; color: #aaf0ca; }
    .ascii { margin: 0; padding: 14px 18px; overflow: auto; border-bottom: 1px solid #2b3035; color: #cfe8ff; background: #11161a; font: 15px/1.15 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; letter-spacing: 1px; white-space: pre; }
    pre { margin: 0; padding: 18px; overflow: auto; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; }
  </style>
</head>
<body>
  <header>
    <h1>GVGAI Live Metadata Preview</h1>
    <div id="status">connecting...</div>
  </header>
  <main>
    <section class="stage"><img id="frame" alt="GVGAI frame"></section>
    <section class="meta">
      <div class="chips" id="chips"></div>
      <pre class="ascii" id="ascii"></pre>
      <pre id="metadata">{}</pre>
    </section>
  </main>
  <script>
    const frame = document.getElementById("frame");
    const metadata = document.getElementById("metadata");
    const ascii = document.getElementById("ascii");
    const status = document.getElementById("status");
    const chips = document.getElementById("chips");

    function chip(label, active=false) {
      const span = document.createElement("span");
      span.className = "chip" + (active ? " active" : "");
      span.textContent = label;
      return span;
    }

    async function refresh() {
      frame.src = "/frame?ts=" + Date.now();
      const res = await fetch("/state?ts=" + Date.now(), { cache: "no-store" });
      const state = await res.json();
      status.textContent = state.status + " | frames " + state.frame_count + (state.error ? " | " + state.error : "");
      const meta = state.metadata || {};
      const variants = meta.variants || {};
      const active = new Set(variants.active || []);
      chips.replaceChildren(
        chip("game: " + (meta.game || "")),
        chip("variant: " + (meta.variant || "")),
        chip("action: " + ((meta.action && meta.action.semantic) || "")),
        ...(variants.configured || []).map(v => chip(v, active.has(v)))
      );
      const asciiObj = state.ascii || {};
      const legend = asciiObj.legend ? "legend " + JSON.stringify(asciiObj.legend) : "";
      ascii.textContent = (asciiObj.grid || "") + (legend ? "\\n\\n" + legend : "");
      metadata.textContent = JSON.stringify(meta, null, 2);
    }
    refresh();
    setInterval(refresh, 250);
  </script>
</body>
</html>
"""


class PreviewHandler(BaseHTTPRequestHandler):
	state: LiveState

	def log_message(self, format: str, *args: Any) -> None:
		return

	def do_GET(self) -> None:
		path = urlparse(self.path).path
		if path == "/":
			self._send_bytes(html_page(), "text/html; charset=utf-8")
		elif path == "/state":
			self._send_bytes(
				json.dumps(self.state.snapshot(), sort_keys=True).encode("utf-8"),
				"application/json; charset=utf-8",
			)
		elif path == "/frame":
			image = self.state.image()
			if not image:
				image = as_png(np.zeros((64, 64, 3), dtype=np.uint8))
			self._send_bytes(image, "image/png")
		else:
			self.send_error(HTTPStatus.NOT_FOUND)

	def _send_bytes(self, payload: bytes, content_type: str) -> None:
		self.send_response(HTTPStatus.OK)
		self.send_header("Content-Type", content_type)
		self.send_header("Cache-Control", "no-store")
		self.send_header("Content-Length", str(len(payload)))
		self.end_headers()
		self.wfile.write(payload)


def run_rollout(args: argparse.Namespace, state: LiveState) -> None:
	try:
		jobs = load_jobs(args.catalog, parse_filter(args.game), parse_filter(args.variant))
		if len(jobs) != 1:
			keys = ", ".join(job.env_key for job in jobs[:8])
			raise ValueError(f"preview needs exactly one job, got {len(jobs)}: {keys}")
		job = jobs[0]
		rng = random.Random(args.seed)
		labeler = FrameLabeler(job.rule_flags, job.variant_label_rules)
		episode_id = 0

		while True:
			level = job.levels[episode_id % len(job.levels)] if args.level is None else args.level
			episode_seed = rng.randrange(2**31 - 1)
			env = make_env(job, level=level, seed=episode_seed, gvgai_root=args.gvgai_root, max_episode_steps=args.max_episode_steps)
			try:
				obs, info_t = env.reset(seed=episode_seed, options={"level": level})
				labeler.reset_episode()
				last_action: int | None = None
				step_in_episode = 0
				done = False

				while not done:
					action = choose_action(args.policy, env.action_space, rng, last_action)
					next_obs, _reward, terminated, truncated, info_next = env.step(action)
					action_meanings = (
						env.get_action_meanings()
						if hasattr(env, "get_action_meanings")
						else list(info_next.get("actions", []))
					)
					_, semantic_action = action_semantic(int(action), action_meanings)
					objects_t, objects_next, classified_events = prepare_frame_inputs(info_t, info_next)
					active_variants = labeler.active_variants(
						frame_index=state.frame_count,
						action=semantic_action,
						objects_t=objects_t,
						objects_next=objects_next,
						classified_events=classified_events,
					)
					metadata = build_frame_metadata(
						game=job.game,
						rollout_variant=job.variant,
						level=int(level),
						episode_id=int(episode_id),
						step_in_episode=int(step_in_episode),
						seed=int(episode_seed),
						action_id=int(action),
						action_meanings=action_meanings,
						info_t=info_t,
						info_next=info_next,
						static_rule_flags=job.rule_flags,
						active_variants=active_variants,
						classified_events=classified_events,
						objects_t=objects_t,
					)
					state.update(as_png(next_obs), metadata)
					obs = next_obs
					info_t = info_next
					last_action = int(action)
					step_in_episode += 1
					done = bool(terminated or truncated)
					time.sleep(max(0.0, args.step_delay))
			finally:
				env.close()
			episode_id += 1
	except BaseException as exc:
		state.fail(exc)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Serve a live GVGAI frame plus frame-metadata preview.")
	parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
	parser.add_argument("--game", default="aliens", help="Single game name.")
	parser.add_argument("--variant", default="physics_a", help="Single variant name.")
	parser.add_argument("--level", type=int, default=0)
	parser.add_argument("--policy", choices=["random", "repeat_random", "noop"], default="repeat_random")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--gvgai-root", type=Path, default=None)
	parser.add_argument("--max-episode-steps", type=int, default=2_000)
	parser.add_argument("--step-delay", type=float, default=0.2)
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--port", type=int, default=8765)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	state = LiveState()
	PreviewHandler.state = state
	thread = threading.Thread(target=run_rollout, args=(args, state), daemon=True)
	thread.start()
	server = ThreadingHTTPServer((args.host, args.port), PreviewHandler)
	print(f"Serving GVGAI live preview on http://{args.host}:{args.port}", flush=True)
	server.serve_forever()


if __name__ == "__main__":
	main()
