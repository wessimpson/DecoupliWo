from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


FRAME_METADATA_SCHEMA_VERSION = "gvgai-frame-metadata-v1"


ACTION_SEMANTICS = {
	"ACTION_NIL": "idle",
	"ACTION_LEFT": "move_left",
	"ACTION_RIGHT": "move_right",
	"ACTION_UP": "move_up",
	"ACTION_DOWN": "move_down",
	"ACTION_USE": "fire_projectile",
}

STATE_GROUPS = {
	"avatar": "avatar",
	"NPCPositions": "enemy",
	"immovablePositions": "immovable",
	"movablePositions": "movable",
	"resourcesPositions": "resource",
	"portalsPositions": "portal",
	"fromAvatarSpritesPositions": "player_projectile",
}

PROJECTILE_VARIANT_HINTS = {
	"multishot",
	"split_orthogonal",
	"ricochet",
	"shoot_walls",
}

HIT_VARIANT_HINTS = {
	"enemy_explode",
	"two_hit_color",
}

GLOBAL_VARIANT_HINTS = {
	"physics",
	"gravity",
	"left_is_right",
	"right_is_left",
	"up_is_down",
	"down_is_up",
	"control",
	"disturbance",
}

ASCII_PRIORITY = [
	("avatar", "@"),
	("sam", "|"),
	("missile", "|"),
	("projectile", "|"),
	("bullet", "|"),
	("shot", "|"),
	("bomb", "*"),
	("alien", "E"),
	("enemy", "E"),
	("base", "#"),
	("wall", "#"),
	("shield", "#"),
	("portal", "O"),
	("resource", "$"),
]

ASCII_FALLBACK_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"


def action_semantic(action_id: int, action_meanings: list[str]) -> tuple[str, str]:
	raw = action_meanings[action_id] if 0 <= action_id < len(action_meanings) else str(action_id)
	return raw, ACTION_SEMANTICS.get(raw, raw.lower())


def _loads_json(raw: Any) -> dict[str, Any]:
	if isinstance(raw, dict):
		return raw
	if raw is None:
		return {}
	try:
		value = json.loads(str(raw))
	except Exception:
		return {}
	return value if isinstance(value, dict) else {}


def _position(raw: Any) -> list[float] | None:
	if raw is None:
		return None
	if isinstance(raw, dict):
		if "x" in raw and "y" in raw:
			return [float(raw["x"]), float(raw["y"])]
		if "0" in raw and "1" in raw:
			return [float(raw["0"]), float(raw["1"])]
	if isinstance(raw, (list, tuple)) and len(raw) >= 2:
		return [float(raw[0]), float(raw[1])]
	return None


def _same_position(left: Any, right: Any) -> bool:
	left_pos = _position(left)
	right_pos = _position(right)
	if left_pos is None or right_pos is None:
		return False
	return abs(left_pos[0] - right_pos[0]) < 1e-6 and abs(left_pos[1] - right_pos[1]) < 1e-6


def _char_for_token(token: str, assigned: dict[str, str]) -> str:
	normalized = token.lower()
	for fragment, char in ASCII_PRIORITY:
		if fragment in normalized:
			return char
	if token not in assigned:
		used = set(assigned.values()) | {char for _, char in ASCII_PRIORITY} | {"."}
		assigned[token] = next((char for char in ASCII_FALLBACK_CHARS if char not in used), "?")
	return assigned[token]


def compact_ascii(raw_ascii: Any) -> dict[str, Any]:
	"""Convert GVGAI's comma/cell token grid into a one-character-per-cell grid."""
	raw = str(raw_ascii or "")
	legend: dict[str, set[str]] = {".": {"empty"}}
	assigned: dict[str, str] = {}
	grid_lines: list[str] = []

	for raw_line in raw.splitlines():
		cells = raw_line.split(",")
		chars: list[str] = []
		for cell in cells:
			tokens = [token for token in cell.split() if token]
			if not tokens:
				char = "."
			else:
				token_chars = [(_char_for_token(token, assigned), token) for token in tokens]
				char, _token = min(
					token_chars,
					key=lambda item: ASCII_PRIORITY.index(next(
						priority for priority in ASCII_PRIORITY if priority[1] == item[0]
					)) if any(priority[1] == item[0] for priority in ASCII_PRIORITY) else len(ASCII_PRIORITY),
				)
				for token_char, token in token_chars:
					legend.setdefault(token_char, set()).add(token)
			chars.append(char)
		grid_lines.append("".join(chars))

	return {
		"encoding": "single_char_grid_v1",
		"grid": "\n".join(grid_lines),
		"raw": raw,
		"legend": {char: sorted(tokens) for char, tokens in sorted(legend.items())},
		"empty": ".",
	}


def _walk_observations(raw: Any):
	if isinstance(raw, dict):
		if "itype" in raw or "itypeKey" in raw or "obsID" in raw:
			yield raw
			return
		for value in raw.values():
			yield from _walk_observations(value)
	elif isinstance(raw, list):
		for value in raw:
			yield from _walk_observations(value)


def _compact_observation(raw: dict[str, Any], role: str) -> dict[str, Any]:
	return {
		"role": role,
		"type_id": int(raw.get("itype", -1)),
		"type_key": str(raw.get("itypeKey") or ""),
		"sprite_id": int(raw.get("obsID", -1)),
		"position": _position(raw.get("position")),
		"category": int(raw.get("category", -1)),
	}


def extract_state_objects(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
	objects: dict[str, list[dict[str, Any]]] = {role: [] for role in set(STATE_GROUPS.values())}

	avatar_position = _position(state.get("avatarPosition"))
	if avatar_position is not None:
		objects["avatar"].append({
			"role": "avatar",
			"type_id": int(state.get("avatarType", -1)),
			"type_key": "",
			"sprite_id": -1,
			"position": avatar_position,
			"category": -1,
		})

	for group_key, role in STATE_GROUPS.items():
		if group_key == "avatar":
			continue
		for raw in _walk_observations(state.get(group_key)):
			objects.setdefault(role, []).append(_compact_observation(raw, role))
	return objects


def type_roles(objects: dict[str, list[dict[str, Any]]]) -> dict[int, set[str]]:
	roles: dict[int, set[str]] = {}
	for role, rows in objects.items():
		for row in rows:
			type_id = int(row.get("type_id", -1))
			if type_id >= 0:
				roles.setdefault(type_id, set()).add(role)
	return roles


def _merge_type_roles(*role_maps: dict[int, set[str]]) -> dict[int, set[str]]:
	merged: dict[int, set[str]] = {}
	for role_map in role_maps:
		for type_id, roles in role_map.items():
			merged.setdefault(type_id, set()).update(roles)
	return merged


def _event_role(type_id: int, roles_by_type: dict[int, set[str]]) -> str:
	roles = roles_by_type.get(int(type_id), set())
	if "avatar" in roles:
		return "player"
	if "player_projectile" in roles:
		return "player_projectile"
	if "enemy" in roles:
		return "enemy"
	if "immovable" in roles:
		return "barrier"
	if "movable" in roles:
		return "movable"
	return "unknown"


def classify_events(events: list[dict[str, Any]], roles_by_type: dict[int, set[str]]) -> list[dict[str, Any]]:
	classified: list[dict[str, Any]] = []
	for event in events:
		active_type = int(event.get("active_type_id", -1))
		passive_type = int(event.get("passive_type_id", -1))
		active_role = _event_role(active_type, roles_by_type)
		passive_role = _event_role(passive_type, roles_by_type)
		labels: list[str] = []

		if bool(event.get("from_avatar")) and passive_role == "enemy":
			labels.append("enemy.hit_by_projectile")
		if active_role == "player" and passive_role in {"barrier", "immovable"}:
			labels.append("player.object_collision")
		if active_role == "player" and passive_role not in {"barrier", "unknown"}:
			labels.append(f"player.touched_{passive_role}")
		if active_role in {"enemy", "movable", "unknown"} and passive_role == "player":
			labels.append("player.hit_by_projectile")

		classified.append({
			"game_step": int(event.get("game_step", -1)),
			"from_avatar": bool(event.get("from_avatar", False)),
			"active": {
				"type_id": active_type,
				"type_key": str(event.get("active_type_key") or ""),
				"sprite_id": int(event.get("active_sprite_id", -1)),
				"role": active_role,
			},
			"passive": {
				"type_id": passive_type,
				"type_key": str(event.get("passive_type_key") or ""),
				"sprite_id": int(event.get("passive_sprite_id", -1)),
				"role": passive_role,
			},
			"position": _position(event.get("position")),
			"labels": labels,
		})
	return classified


def _enabled_rule_names(static_rule_flags: dict[str, float | int | bool]) -> set[str]:
	enabled: set[str] = set()
	for name, value in static_rule_flags.items():
		try:
			is_enabled = float(value) != 0.0
		except (TypeError, ValueError):
			is_enabled = bool(value)
		if is_enabled:
			enabled.add(name)
	return enabled


def _has_projectile_window(variant: str, rule: dict[str, Any] | None) -> bool:
	if rule and rule.get("activation") == "player_projectile_window":
		return True
	return any(hint in variant for hint in PROJECTILE_VARIANT_HINTS)


def _has_hit_window(variant: str, rule: dict[str, Any] | None) -> bool:
	if rule and rule.get("activation") == "enemy_hit_window":
		return True
	return any(hint in variant for hint in HIT_VARIANT_HINTS)


def _always_active(rule: dict[str, Any] | None) -> bool:
	return bool(rule and rule.get("activation") == "always")


def _default_always_active(variant: str, rule: dict[str, Any] | None) -> bool:
	if rule:
		return False
	return any(hint in variant for hint in GLOBAL_VARIANT_HINTS)


@dataclass
class FrameLabeler:
	static_rule_flags: dict[str, float | int | bool]
	variant_label_rules: dict[str, dict[str, Any]] = field(default_factory=dict)
	default_projectile_grace: int = 0
	default_hit_grace: int = 4
	_active_until: dict[str, int] = field(default_factory=dict)

	def reset_episode(self) -> None:
		self._active_until.clear()

	def active_variants(
		self,
		frame_index: int,
		action: str,
		objects_t: dict[str, list[dict[str, Any]]],
		objects_next: dict[str, list[dict[str, Any]]],
		classified_events: list[dict[str, Any]],
	) -> set[str]:
		enabled = _enabled_rule_names(self.static_rule_flags)
		active = {
			name
			for name, until in self._active_until.items()
			if name in enabled and until >= frame_index
		}
		player_projectiles_visible = bool(objects_t.get("player_projectile") or objects_next.get("player_projectile"))
		enemy_hit = any("enemy.hit_by_projectile" in event["labels"] for event in classified_events)

		for variant in enabled:
			rule = self.variant_label_rules.get(variant)
			if _always_active(rule) or _default_always_active(variant, rule):
				active.add(variant)
				continue
			if _has_projectile_window(variant, rule) and (action == "fire_projectile" or player_projectiles_visible):
				grace = int(rule.get("grace_frames", self.default_projectile_grace)) if rule else self.default_projectile_grace
				self._active_until[variant] = max(self._active_until.get(variant, -1), frame_index + grace)
				active.add(variant)
			if _has_hit_window(variant, rule) and enemy_hit:
				grace = int(rule.get("grace_frames", self.default_hit_grace)) if rule else self.default_hit_grace
				self._active_until[variant] = max(self._active_until.get(variant, -1), frame_index + grace)
				active.add(variant)
		return active


def build_frame_metadata(
	*,
	game: str,
	rollout_variant: str,
	level: int,
	episode_id: int,
	step_in_episode: int,
	seed: int,
	action_id: int,
	action_meanings: list[str],
	info_t: dict[str, Any],
	info_next: dict[str, Any],
	static_rule_flags: dict[str, float | int | bool],
	active_variants: set[str],
	classified_events: list[dict[str, Any]],
	objects_t: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
	raw_action, semantic_action = action_semantic(action_id, action_meanings)
	active_flags = {name: float(name in active_variants) for name in sorted(static_rule_flags)}
	player_receiving = sorted({
		label.split(".", 1)[1]
		for event in classified_events
		for label in event["labels"]
		if label.startswith("player.")
	})
	if (
		semantic_action in {"move_left", "move_right", "move_up", "move_down"}
		and _same_position(info_t.get("avatar_xy"), info_next.get("avatar_xy"))
		and "object_collision" not in player_receiving
	):
		player_receiving.append("object_collision")
	enemy_receiving = sorted({
		label.split(".", 1)[1]
		for event in classified_events
		for label in event["labels"]
		if label.startswith("enemy.")
	})

	return {
		"schema_version": FRAME_METADATA_SCHEMA_VERSION,
		"game": game,
		"level": int(level),
		"rollout_variant": rollout_variant,
		"configured_variants": sorted(_enabled_rule_names(static_rule_flags)),
		"active_variants": sorted(active_variants),
		"episode_id": int(episode_id),
		"step_in_episode": int(step_in_episode),
		"seed": int(seed),
		"tick": int(info_t.get("game_tick", -1)),
		"next_tick": int(info_next.get("game_tick", -1)),
		"action": {
			"id": int(action_id),
			"raw": raw_action,
			"semantic": semantic_action,
		},
		"labels": {
			"player": {
				"making": semantic_action,
				"receiving": player_receiving,
			},
			"enemy": {
				"making": [],
				"receiving": enemy_receiving,
			},
			"global_disturbance": [],
		},
		"frame": {
			"ascii": compact_ascii(info_t.get("ascii", "")),
			"score": float(info_t.get("score", 0.0)),
			"winner": str(info_t.get("winner", "NO_WINNER")),
			"block_size": int(info_t.get("block_size", 0)),
			"objects": objects_t,
		},
		"events": classified_events,
		"rule_flags_static": {name: float(value) for name, value in sorted(static_rule_flags.items())},
		"rule_flags_active": active_flags,
	}


def prepare_frame_inputs(info_t: dict[str, Any], info_next: dict[str, Any]):
	state_t = _loads_json(info_t.get("observation_json"))
	state_next = _loads_json(info_next.get("observation_json"))
	objects_t = extract_state_objects(state_t)
	objects_next = extract_state_objects(state_next)
	roles_by_type = _merge_type_roles(type_roles(objects_t), type_roles(objects_next))
	classified_events = classify_events(list(info_next.get("events") or []), roles_by_type)
	return objects_t, objects_next, classified_events
