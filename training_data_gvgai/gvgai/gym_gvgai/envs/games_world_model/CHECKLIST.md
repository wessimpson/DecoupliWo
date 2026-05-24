# games_world_model checklist

Track verification of every world-model game and rule variant.

## Summary

| Game | Variants | Levels |
|---|---:|---:|
| aliens | 19 | 5 |
| chopper | 19 | 5 |
| defender | 13 | 5 |
| frogs | 5 | 6 |
| ikaruga | 13 | 6 |
| jaws | 14 | 5 |
| missilecommand | 10 | 6 |
| pacman | 7 | 6 |
| roadfighter | 6 | 6 |
| seaquest | 13 | 6 |
| sheriff | 13 | 6 |
| waves | 21 | 5 |
| zelda | 7 | 5 |
| **Total** | **160** | |

## How to verify

Mark **rule working** when the variant mechanic behaves as intended in play.
Mark **agent working** when MCTS/random can run episodes without Java errors or crashes.

```powershell
python training_data_gvgai/run_mcts.py --env <game> --rules <tag> --level 0 --show
python training_data_gvgai/run_random_action.py --env <game> --rules <tag> --level 0 --show
# base game: omit --rules
```

### Rule families

| Family | Tags |
|---|---|
| **Ikaruga-style** | `big_explosion_3rad`, `big_shot_*`, `multishot_2/3/5`, `pierce_shot`, `quick_dash_*`, `shield_reflect` |
| **Shooter extras** | `enemy_explode_*rad`, `enemy_multishot`, `multishot` (spread), `ricochet`, `shoot_walls`, `split_orthogonal`, `two_hit_color` |
| **Other** | `enemy_speed_2x`, `car_speed_2x`, `oil_slowdown`, `ghost_*`, `wall_on_death`, etc. |

---

## aliens

*5 shared level(s): `lvl0` … `lvl4`*

| Variant | `--rules` tag | rule working | agent working |
|---|---|:---:|:---:|
| `aliens` | *(omit)* | [ ] | [ ] |
| `aliens_rules_big_shot_2x` | `big_shot_2x` | [ ] | [ ] |
| `aliens_rules_big_shot_4x` | `big_shot_4x` | [ ] | [ ] |
| `aliens_rules_multishot_2` | `multishot_2` | [ ] | [ ] |
| `aliens_rules_multishot_3` | `multishot_3` | [ ] | [ ] |
| `aliens_rules_multishot_5` | `multishot_5` | [ ] | [ ] |
| `aliens_rules_pierce_shot` | `pierce_shot` | [ ] | [ ] |
| `aliens_rules_quick_dash_2tile` | `quick_dash_2tile` | [ ] | [ ] |
| `aliens_rules_quick_dash_3tile` | `quick_dash_3tile` | [ ] | [ ] |
| `aliens_rules_quick_dash_5tile` | `quick_dash_5tile` | [ ] | [ ] |
| `aliens_rules_shield_reflect` | `shield_reflect` | [ ] | [ ] |
| *game-specific* | | | |
| `aliens_rules_enemy_explode_1rad` | `enemy_explode_1rad` | [ ] | [ ] |
| `aliens_rules_enemy_explode_3rad` | `enemy_explode_3rad` | [ ] | [ ] |
| `aliens_rules_enemy_explode_5rad` | `enemy_explode_5rad` | [ ] | [ ] |
| `aliens_rules_enemy_multishot` | `enemy_multishot` | [ ] | [ ] |
| `aliens_rules_ricochet` | `ricochet` | [ ] | [ ] |
| `aliens_rules_shoot_walls` | `shoot_walls` | [ ] | [ ] |
| `aliens_rules_split_orthogonal` | `split_orthogonal` | [ ] | [ ] |
| `aliens_rules_two_hit_color` | `two_hit_color` | [ ] | [ ] |

## chopper

*5 shared level(s): `lvl0` … `lvl4`*

| Variant | `--rules` tag | rule working | agent working |
|---|---|:---:|:---:|
| `chopper` | *(omit)* | [ ] | [ ] |
| `chopper_rules_big_shot_1x` | `big_shot_1x` | [ ] | [ ] |
| `chopper_rules_big_shot_2x` | `big_shot_2x` | [ ] | [ ] |
| `chopper_rules_big_shot_4x` | `big_shot_4x` | [ ] | [ ] |
| `chopper_rules_multishot_2` | `multishot_2` | [ ] | [ ] |
| `chopper_rules_multishot_3` | `multishot_3` | [ ] | [ ] |
| `chopper_rules_multishot_5` | `multishot_5` | [ ] | [ ] |
| `chopper_rules_pierce_shot` | `pierce_shot` | [ ] | [ ] |
| `chopper_rules_quick_dash_2tile` | `quick_dash_2tile` | [ ] | [ ] |
| `chopper_rules_quick_dash_3tile` | `quick_dash_3tile` | [ ] | [ ] |
| `chopper_rules_quick_dash_5tile` | `quick_dash_5tile` | [ ] | [ ] |
| `chopper_rules_shield_reflect` | `shield_reflect` | [ ] | [ ] |
| *game-specific* | | | |
| `chopper_rules_enemy_explode_1rad` | `enemy_explode_1rad` | [ ] | [ ] |
| `chopper_rules_enemy_explode_3rad` | `enemy_explode_3rad` | [ ] | [ ] |
| `chopper_rules_enemy_explode_5rad` | `enemy_explode_5rad` | [ ] | [ ] |
| `chopper_rules_enemy_multishot` | `enemy_multishot` | [ ] | [ ] |
| `chopper_rules_multishot` | `multishot` | [ ] | [ ] |
| `chopper_rules_ricochet` | `ricochet` | [ ] | [ ] |
| `chopper_rules_split_orthogonal` | `split_orthogonal` | [ ] | [ ] |

## defender

*5 shared level(s): `lvl0` … `lvl4`*

| Variant | `--rules` tag | rule working | agent working |
|---|---|:---:|:---:|
| `defender` | *(omit)* | [ ] | [ ] |
| `defender_rules_big_shot_1x` | `big_shot_1x` | [ ] | [ ] |
| `defender_rules_big_shot_2x` | `big_shot_2x` | [ ] | [ ] |
| `defender_rules_big_shot_4x` | `big_shot_4x` | [ ] | [ ] |
| `defender_rules_multishot_2` | `multishot_2` | [ ] | [ ] |
| `defender_rules_multishot_3` | `multishot_3` | [ ] | [ ] |
| `defender_rules_multishot_5` | `multishot_5` | [ ] | [ ] |
| `defender_rules_pierce_shot` | `pierce_shot` | [ ] | [ ] |
| `defender_rules_quick_dash_2tile` | `quick_dash_2tile` | [ ] | [ ] |
| `defender_rules_quick_dash_3tile` | `quick_dash_3tile` | [ ] | [ ] |
| `defender_rules_quick_dash_5tile` | `quick_dash_5tile` | [ ] | [ ] |
| `defender_rules_shield_reflect` | `shield_reflect` | [ ] | [ ] |
| *game-specific* | | | |
| `defender_rules_enemy_speed_2x` | `enemy_speed_2x` | [ ] | [ ] |

## frogs

*6 shared level(s): `lvl0` … `lvl5`*

| Variant | `--rules` tag | rule working | agent working |
|---|---|:---:|:---:|
| `frogs` | *(omit)* | [ ] | [ ] |
| `frogs_rules_quick_dash_2tile` | `quick_dash_2tile` | [ ] | [ ] |
| `frogs_rules_quick_dash_3tile` | `quick_dash_3tile` | [ ] | [ ] |
| `frogs_rules_quick_dash_5tile` | `quick_dash_5tile` | [ ] | [ ] |
| *game-specific* | | | |
| `frogs_rules_car_speed_2x` | `car_speed_2x` | [ ] | [ ] |

## ikaruga

*6 shared level(s): `lvl0` … `lvl5`*

| Variant | `--rules` tag | rule working | agent working |
|---|---|:---:|:---:|
| `ikaruga` | *(omit)* | [ ] | [ ] |
| `ikaruga_rules_big_explosion_3rad` | `big_explosion_3rad` | [ ] | [ ] |
| `ikaruga_rules_big_shot_1x` | `big_shot_1x` | [ ] | [ ] |
| `ikaruga_rules_big_shot_2x` | `big_shot_2x` | [ ] | [ ] |
| `ikaruga_rules_big_shot_4x` | `big_shot_4x` | [ ] | [ ] |
| `ikaruga_rules_multishot_2` | `multishot_2` | [ ] | [ ] |
| `ikaruga_rules_multishot_3` | `multishot_3` | [ ] | [ ] |
| `ikaruga_rules_multishot_5` | `multishot_5` | [ ] | [ ] |
| `ikaruga_rules_pierce_shot` | `pierce_shot` | [ ] | [ ] |
| `ikaruga_rules_quick_dash_2tile` | `quick_dash_2tile` | [ ] | [ ] |
| `ikaruga_rules_quick_dash_3tile` | `quick_dash_3tile` | [ ] | [ ] |
| `ikaruga_rules_quick_dash_5tile` | `quick_dash_5tile` | [ ] | [ ] |
| `ikaruga_rules_shield_reflect` | `shield_reflect` | [ ] | [ ] |

## jaws

*5 shared level(s): `lvl0` … `lvl4`*

| Variant | `--rules` tag | rule working | agent working |
|---|---|:---:|:---:|
| `jaws` | *(omit)* | [ ] | [ ] |
| `jaws_rules_big_explosion_3rad` | `big_explosion_3rad` | [ ] | [ ] |
| `jaws_rules_big_shot_1x` | `big_shot_1x` | [ ] | [ ] |
| `jaws_rules_big_shot_2x` | `big_shot_2x` | [ ] | [ ] |
| `jaws_rules_big_shot_4x` | `big_shot_4x` | [ ] | [ ] |
| `jaws_rules_multishot_2` | `multishot_2` | [ ] | [ ] |
| `jaws_rules_multishot_3` | `multishot_3` | [ ] | [ ] |
| `jaws_rules_multishot_5` | `multishot_5` | [ ] | [ ] |
| `jaws_rules_pierce_shot` | `pierce_shot` | [ ] | [ ] |
| `jaws_rules_quick_dash_2tile` | `quick_dash_2tile` | [ ] | [ ] |
| `jaws_rules_quick_dash_3tile` | `quick_dash_3tile` | [ ] | [ ] |
| `jaws_rules_quick_dash_5tile` | `quick_dash_5tile` | [ ] | [ ] |
| `jaws_rules_shield_reflect` | `shield_reflect` | [ ] | [ ] |
| *game-specific* | | | |
| `jaws_rules_enemy_speed_2x` | `enemy_speed_2x` | [ ] | [ ] |

## missilecommand

*6 shared level(s): `lvl0` … `lvl5`*

| Variant | `--rules` tag | rule working | agent working |
|---|---|:---:|:---:|
| `missilecommand` | *(omit)* | [ ] | [ ] |
| `missilecommand_rules_big_explosion_3rad` | `big_explosion_3rad` | [ ] | [ ] |
| `missilecommand_rules_big_shot_1x` | `big_shot_1x` | [ ] | [ ] |
| `missilecommand_rules_big_shot_2x` | `big_shot_2x` | [ ] | [ ] |
| `missilecommand_rules_big_shot_4x` | `big_shot_4x` | [ ] | [ ] |
| `missilecommand_rules_multishot_2` | `multishot_2` | [ ] | [ ] |
| `missilecommand_rules_multishot_3` | `multishot_3` | [ ] | [ ] |
| `missilecommand_rules_multishot_5` | `multishot_5` | [ ] | [ ] |
| `missilecommand_rules_pierce_shot` | `pierce_shot` | [ ] | [ ] |
| *game-specific* | | | |
| `missilecommand_rules_big_explosion_5rad` | `big_explosion_5rad` | [ ] | [ ] |

## pacman

*6 shared level(s): `lvl0` … `lvl5`*

| Variant | `--rules` tag | rule working | agent working |
|---|---|:---:|:---:|
| `pacman` | *(omit)* | [ ] | [ ] |
| `pacman_rules_quick_dash_2tile` | `quick_dash_2tile` | [ ] | [ ] |
| `pacman_rules_quick_dash_3tile` | `quick_dash_3tile` | [ ] | [ ] |
| `pacman_rules_quick_dash_5tile` | `quick_dash_5tile` | [ ] | [ ] |
| *game-specific* | | | |
| `pacman_rules_ghost_freeze_on_powerup` | `ghost_freeze_on_powerup` | [ ] | [ ] |
| `pacman_rules_ghost_speed_2x` | `ghost_speed_2x` | [ ] | [ ] |
| `pacman_rules_wall_on_death` | `wall_on_death` | [ ] | [ ] |

## roadfighter

*6 shared level(s): `lvl0` … `lvl5`*

| Variant | `--rules` tag | rule working | agent working |
|---|---|:---:|:---:|
| `roadfighter` | *(omit)* | [ ] | [ ] |
| `roadfighter_rules_quick_dash_2tile` | `quick_dash_2tile` | [ ] | [ ] |
| `roadfighter_rules_quick_dash_3tile` | `quick_dash_3tile` | [ ] | [ ] |
| `roadfighter_rules_quick_dash_5tile` | `quick_dash_5tile` | [ ] | [ ] |
| *game-specific* | | | |
| `roadfighter_rules_car_speed_2x` | `car_speed_2x` | [ ] | [ ] |
| `roadfighter_rules_oil_slowdown` | `oil_slowdown` | [ ] | [ ] |

## seaquest

*6 shared level(s): `lvl0` … `lvl5`*

| Variant | `--rules` tag | rule working | agent working |
|---|---|:---:|:---:|
| `seaquest` | *(omit)* | [ ] | [ ] |
| `seaquest_rules_big_shot_1x` | `big_shot_1x` | [ ] | [ ] |
| `seaquest_rules_big_shot_2x` | `big_shot_2x` | [ ] | [ ] |
| `seaquest_rules_big_shot_4x` | `big_shot_4x` | [ ] | [ ] |
| `seaquest_rules_multishot_2` | `multishot_2` | [ ] | [ ] |
| `seaquest_rules_multishot_3` | `multishot_3` | [ ] | [ ] |
| `seaquest_rules_multishot_5` | `multishot_5` | [ ] | [ ] |
| `seaquest_rules_pierce_shot` | `pierce_shot` | [ ] | [ ] |
| `seaquest_rules_quick_dash_2tile` | `quick_dash_2tile` | [ ] | [ ] |
| `seaquest_rules_quick_dash_3tile` | `quick_dash_3tile` | [ ] | [ ] |
| `seaquest_rules_quick_dash_5tile` | `quick_dash_5tile` | [ ] | [ ] |
| `seaquest_rules_shield_reflect` | `shield_reflect` | [ ] | [ ] |
| *game-specific* | | | |
| `seaquest_rules_enemy_speed_2x` | `enemy_speed_2x` | [ ] | [ ] |

## sheriff

*6 shared level(s): `lvl0` … `lvl5`*

| Variant | `--rules` tag | rule working | agent working |
|---|---|:---:|:---:|
| `sheriff` | *(omit)* | [ ] | [ ] |
| `sheriff_rules_big_explosion_3rad` | `big_explosion_3rad` | [ ] | [ ] |
| `sheriff_rules_big_shot_1x` | `big_shot_1x` | [ ] | [ ] |
| `sheriff_rules_big_shot_2x` | `big_shot_2x` | [ ] | [ ] |
| `sheriff_rules_big_shot_4x` | `big_shot_4x` | [ ] | [ ] |
| `sheriff_rules_multishot_2` | `multishot_2` | [ ] | [ ] |
| `sheriff_rules_multishot_3` | `multishot_3` | [ ] | [ ] |
| `sheriff_rules_multishot_5` | `multishot_5` | [ ] | [ ] |
| `sheriff_rules_pierce_shot` | `pierce_shot` | [ ] | [ ] |
| `sheriff_rules_quick_dash_2tile` | `quick_dash_2tile` | [ ] | [ ] |
| `sheriff_rules_quick_dash_3tile` | `quick_dash_3tile` | [ ] | [ ] |
| `sheriff_rules_quick_dash_5tile` | `quick_dash_5tile` | [ ] | [ ] |
| *game-specific* | | | |
| `sheriff_rules_wall_on_death` | `wall_on_death` | [ ] | [ ] |

## waves

*5 shared level(s): `lvl0` … `lvl4`*

| Variant | `--rules` tag | rule working | agent working |
|---|---|:---:|:---:|
| `waves` | *(omit)* | [ ] | [ ] |
| `waves_rules_big_shot_1x` | `big_shot_1x` | [ ] | [ ] |
| `waves_rules_big_shot_2x` | `big_shot_2x` | [ ] | [ ] |
| `waves_rules_big_shot_4x` | `big_shot_4x` | [ ] | [ ] |
| `waves_rules_multishot_2` | `multishot_2` | [ ] | [ ] |
| `waves_rules_multishot_3` | `multishot_3` | [ ] | [ ] |
| `waves_rules_multishot_5` | `multishot_5` | [ ] | [ ] |
| `waves_rules_pierce_shot` | `pierce_shot` | [ ] | [ ] |
| `waves_rules_quick_dash_2tile` | `quick_dash_2tile` | [ ] | [ ] |
| `waves_rules_quick_dash_3tile` | `quick_dash_3tile` | [ ] | [ ] |
| `waves_rules_quick_dash_5tile` | `quick_dash_5tile` | [ ] | [ ] |
| `waves_rules_shield_reflect` | `shield_reflect` | [ ] | [ ] |
| *game-specific* | | | |
| `waves_rules_enemy_explode_1rad` | `enemy_explode_1rad` | [ ] | [ ] |
| `waves_rules_enemy_explode_3rad` | `enemy_explode_3rad` | [ ] | [ ] |
| `waves_rules_enemy_explode_5rad` | `enemy_explode_5rad` | [ ] | [ ] |
| `waves_rules_enemy_multishot` | `enemy_multishot` | [ ] | [ ] |
| `waves_rules_multishot` | `multishot` | [ ] | [ ] |
| `waves_rules_ricochet` | `ricochet` | [ ] | [ ] |
| `waves_rules_shoot_walls` | `shoot_walls` | [ ] | [ ] |
| `waves_rules_split_orthogonal` | `split_orthogonal` | [ ] | [ ] |
| `waves_rules_two_hit_color` | `two_hit_color` | [ ] | [ ] |

## zelda

*5 shared level(s): `lvl0` … `lvl4`*

| Variant | `--rules` tag | rule working | agent working |
|---|---|:---:|:---:|
| `zelda` | *(omit)* | [ ] | [ ] |
| `zelda_rules_big_explosion_3rad` | `big_explosion_3rad` | [ ] | [ ] |
| `zelda_rules_quick_dash_2tile` | `quick_dash_2tile` | [ ] | [ ] |
| `zelda_rules_quick_dash_3tile` | `quick_dash_3tile` | [ ] | [ ] |
| `zelda_rules_quick_dash_5tile` | `quick_dash_5tile` | [ ] | [ ] |
| *game-specific* | | | |
| `zelda_rules_enemy_speed_2x` | `enemy_speed_2x` | [ ] | [ ] |
| `zelda_rules_wall_on_death` | `wall_on_death` | [ ] | [ ] |
