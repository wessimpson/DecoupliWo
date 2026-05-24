"""Generate ikaruga-style rule variants for chopper, waves, and jaws."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "gym_gvgai" / "envs" / "games_world_model"

EXPLOSION_SPRITE = "        explosion > Flicker limit=5 singleton=True img=oryx/circleEffect1 shrinkfactor=3\n"

CHOPPER_BASE = """BasicGame square_size=8
    SpriteSet
        layers > Immovable hidden=True
            stratosphere > color=LIGHTBLUE img=oryx/backLBlue
            thermosphere > color=BLUE img=oryx/seaWater
            troposphere > img=oryx/backGrey

        satellite   > RandomNPC    color=WHITE img=newset/satellite cons=1
{avatar_line}
        missile > Missile
{missile_lines}
        cloud  > Missile img=newset/cloud2
            leftCloud  > orientation=LEFT speed=0.2  color=WHITE
            fastLeftCloud  > orientation=LEFT speed=0.8  color=WHITE
            rightCloud  > orientation=RIGHT speed=0.2  color=WHITE
            fastRightCloud  > orientation=RIGHT speed=0.8  color=WHITE

        tank   > Bomber stype=sam  prob=0.1  cooldown=5 speed=0.6 img=newset/tank_blueU
        portal  > SpawnPoint img=oryx/dooropen1 portal=True
            portalBase  > stype=tank  cooldown=40   total=20
            portalAmmo  > stype=supply cooldown=10 prob=0.15

        supply > Missile orientation=LEFT speed=0.25 img=oryx/goldsack shrinkfactor=1.0
        bullet > Resource limit=20
{extra_sprites}
    LevelMapping
        g > avatar stratosphere
        h > satellite thermosphere
        i > portalBase troposphere
        j > portalAmmo stratosphere
        k > thermosphere
        m > stratosphere
        n > troposphere
        o > leftCloud stratosphere
        p > fastLeftCloud stratosphere
        q > rightCloud stratosphere
        r > fastRightCloud stratosphere

    TerminationSet
        SpriteCounter      stype=avatar               limit=0 win=False
        SpriteCounter      stype=satellite               limit=0 win=False
        MultiSpriteCounter stype1=portalBase stype2=tank limit=0 win=True

    InteractionSet
        avatar wall EOS > stepBack
{avatar_sam_lines}
        sam avatar wall EOS > killSprite

{tank_bomb_lines}
{bomb_eos_line}
{bomb_sam_lines}
        tank wall EOS  > reverseDirection
        tank wall EOS  > stepBack

        {satellite_sam_line}
        satellite wall EOS > stepBack

        avatar supply > changeResource resource=bullet value=5  killResource=True
        supply wall EOS > killSprite

        avatar portal thermosphere troposphere > stepBack
        satellite stratosphere > stepBack

        {cloud_sam_line}
        cloud EOS > wrapAround
{extra_interactions}"""

WAVES_BASE = """BasicGame square_size=8
    SpriteSet

        background > Immovable img=oryx/space1 hidden=True
        asteroid > Immovable img=oryx/planet

        missile > Missile
{missile_lines}
        portal  >
            portalSlow  > SpawnPoint   stype=alien  cooldown=10 prob=0.05 img=newset/whirlpool2 portal=True
            rockPortal  > SpawnPoint   stype=rock  cooldown=10 prob=0.2 img=newset/whirlpool1 portal=True

        shield > Resource color=GOLD limit=4 img=oryx/shield2

{avatar_line}
        alien > Bomber color=BROWN img=oryx/alien3 speed=0.2 orientation=LEFT stype=laser prob=0.05
{extra_sprites}
    LevelMapping
        B > background portalSlow
        C > background rockPortal
        D > background avatar shield
        E > background asteroid
        F > background

    TerminationSet
        SpriteCounter      stype=avatar             limit=0 win=False
        Timeout limit=1000 win=True

    InteractionSet
        avatar  EOS  > stepBack
        alien   EOS  > killSprite
        missile EOS  > killSprite

{alien_sam_line}

        sam laser > transformTo stype=shield killSecond=True

        avatar shield > changeResource resource=shield value=1 killResource=True

        avatar rock > killIfHasLess resource=shield limit=0
        avatar rock > changeResource resource=shield value=-1 killResource=True

        avatar alien > killIfHasLess resource=shield limit=0
        avatar alien > changeResource resource=shield value=-1 killResource=True

{avatar_laser_lines}

        asteroid {sam_types} laser > killSprite
        rock asteroid > killSprite
        alien asteroid > killSprite
        laser asteroid > killSprite
        avatar asteroid > stepBack
{extra_interactions}"""

JAWS_BASE = """BasicGame square_size=8
    SpriteSet
        water > Immovable img=newset/water2
        holes > SpawnPoint color=LIGHTGRAY img=newset/whirlpool2 portal=True
            sharkhole  >  stype=shark  prob=0.1 total=1
            whalehole  >  stype=whale  prob=0.05 cooldown=10
            piranhahole  >  stype=piranha  prob=0.05 cooldown=10

        moving >
{avatar_line}
{torpedo_lines}
            fish >
                shark  > Chaser speed=0.2 color=ORANGE img=newset/shark2 stype=avatar
                whale  > Missile  orientation=RIGHT  speed=0.2 color=BROWN img=newset/whale
                piranha > Missile orientation=LEFT speed=0.2 color=RED img=newset/piranha1

        shell > Resource color=GOLD limit=20 img=oryx/amulat1 shrinkfactor=1.0
        sharkFang > Resource color=GOLD limit=1 img=oryx/sword4 shrinkfactor=0.5
{extra_sprites}

    LevelMapping
        1 > water piranhahole
        2 > water whalehole
        3 > water sharkhole
        . > water
        A > water avatar

    TerminationSet
        SpriteCounter stype=avatar limit=0 win=False
        Timeout limit=1000 win=True

    InteractionSet
        EOS avatar shark > stepBack
        EOS {torpedo_types} fish > killSprite

{fish_torpedo_lines}
{torpedo_fish_line}

        sharkFang avatar > collectResource scoreChange=1000
        shell avatar > collectResource scoreChange=1

        avatar shark > spawnIfHasMore resource=shell limit=15 stype=sharkFang
        shark avatar > killIfOtherHasMore resource=shell limit=15

        avatar shark  > killIfHasLess resource=shell limit=15
{avatar_fish_lines}
{extra_interactions}"""


def write_chopper(name: str, **kwargs) -> None:
    text = CHOPPER_BASE.format(
        avatar_line=kwargs["avatar_line"],
        missile_lines=kwargs["missile_lines"],
        extra_sprites=kwargs.get("extra_sprites", ""),
        avatar_sam_lines=kwargs.get("avatar_sam_lines", "        avatar sam > killSprite"),
        tank_bomb_lines=kwargs["tank_bomb_lines"],
        bomb_eos_line=kwargs["bomb_eos_line"],
        bomb_sam_lines=kwargs.get("bomb_sam_lines", "        bomb sam > killBoth\n"),
        satellite_sam_line=kwargs.get(
            "satellite_sam_line", "satellite sam > killBoth scoreChange=-1"
        ),
        cloud_sam_line=kwargs.get("cloud_sam_line", "cloud sam > killBoth"),
        extra_interactions=kwargs.get("extra_interactions", ""),
    )
    (ROOT / "chopper" / f"chopper_rules_{name}.txt").write_text(text, encoding="utf-8")


def write_waves(name: str, **kwargs) -> None:
    text = WAVES_BASE.format(
        missile_lines=kwargs["missile_lines"],
        avatar_line=kwargs["avatar_line"],
        extra_sprites=kwargs.get("extra_sprites", ""),
        alien_sam_line=kwargs["alien_sam_line"],
        avatar_laser_lines=kwargs.get(
            "avatar_laser_lines",
            "        avatar laser > killIfHasLess resource=shield limit=0\n"
            "        avatar laser > changeResource resource=shield value=-1 killResource=True",
        ),
        sam_types=kwargs.get("sam_types", "sam"),
        extra_interactions=kwargs.get("extra_interactions", ""),
    )
    (ROOT / "waves" / f"waves_rules_{name}.txt").write_text(text, encoding="utf-8")


def write_jaws(name: str, **kwargs) -> None:
    text = JAWS_BASE.format(
        avatar_line=kwargs["avatar_line"],
        torpedo_lines=kwargs["torpedo_lines"],
        extra_sprites=kwargs.get("extra_sprites", ""),
        torpedo_types=kwargs.get("torpedo_types", "torpedo"),
        fish_torpedo_lines=kwargs["fish_torpedo_lines"],
        torpedo_fish_line=kwargs.get("torpedo_fish_line", "        torpedo fish > killSprite"),
        avatar_fish_lines=kwargs.get(
            "avatar_fish_lines",
            "        avatar whale piranha > killSprite",
        ),
        extra_interactions=kwargs.get("extra_interactions", ""),
    )
    (ROOT / "jaws" / f"jaws_rules_{name}.txt").write_text(text, encoding="utf-8")


def chopper_avatar(speed: str = "1.0", stype: str = "bomb", spread: str = "") -> str:
    spread_part = f" {spread}" if spread else ""
    return (
        f"        avatar   > ShootAvatar orientation=DOWN color=YELLOW ammo=bullet "
        f"stype={stype} speed={speed}{spread_part} img=newset/helicopter rotateInPlace=False"
    )


def chopper_bomb_line(sf: str) -> str:
    return (
        f"            bomb > orientation=DOWN color=RED speed=0.75 img=newset/bomb "
        f"singleton=True shrinkfactor={sf}"
    )


def chopper_bombs(n: int, sf: str = "0.5") -> str:
    sam = "            sam  > orientation=UP color=BLUE speed=0.25 img=oryx/bullet1 shrinkfactor=2.0\n"
    bombs = "\n".join(
        f"            bomb{i} > orientation=DOWN color=RED speed=0.75 img=newset/bomb shrinkfactor={sf}"
        for i in range(1, n + 1)
    )
    return sam + bombs


def chopper_tank_bomb(stypes: str, effect: str) -> str:
  return f"        tank {stypes} > {effect}"


def gen_chopper() -> None:
    avatar_default = chopper_avatar()
    missile_default = (
        "            sam  > orientation=UP color=BLUE speed=0.25 img=oryx/bullet1 shrinkfactor=2.0\n"
        + chopper_bomb_line("0.5")
    )
    tank_kill = chopper_tank_bomb("bomb", "killSprite scoreChange=1")
    bomb_eos = "        bomb tank wall EOS > killSprite"

    for tag, sf in [("big_shot_1x", "0.5"), ("big_shot_2x", "1.0"), ("big_shot_4x", "2.0")]:
        write_chopper(
            tag,
            avatar_line=avatar_default,
            missile_lines=(
                "            sam  > orientation=UP color=BLUE speed=0.25 img=oryx/bullet1 shrinkfactor=2.0\n"
                + chopper_bomb_line(sf)
            ),
            tank_bomb_lines=tank_kill,
            bomb_eos_line=bomb_eos,
        )

    chopper_multishot_spread = (
        "alignShotToOrientation=False fireAllWeapons=True spreadPixels=0 rotateInPlace=False"
    )
    chopper_multishot_bomb_base = chopper_bomb_line("0.5")
    chopper_multishot_specs = {
        "multishot_2": (
            "bombL,bombR",
            (
                "            sam  > orientation=UP color=BLUE speed=0.25 img=oryx/bullet1 shrinkfactor=2.0\n"
                + chopper_multishot_bomb_base
                + "            bombL > orientation=-0.7071,0.7071 color=RED speed=0.75 img=newset/bomb shrinkfactor=0.5\n"
                "            bombR > orientation=0.7071,0.7071 color=RED speed=0.75 img=newset/bomb shrinkfactor=0.5\n"
            ),
            ["bombL", "bombR"],
        ),
        "multishot_3": (
            "bombL,bombC,bombR",
            (
                "            sam  > orientation=UP color=BLUE speed=0.25 img=oryx/bullet1 shrinkfactor=2.0\n"
                + chopper_multishot_bomb_base
                + "            bombL > orientation=-0.7071,0.7071 color=RED speed=0.75 img=newset/bomb shrinkfactor=0.5\n"
                "            bombC > orientation=DOWN color=RED speed=0.75 img=newset/bomb shrinkfactor=0.5\n"
                "            bombR > orientation=0.7071,0.7071 color=RED speed=0.75 img=newset/bomb shrinkfactor=0.5\n"
            ),
            ["bombL", "bombC", "bombR"],
        ),
        "multishot_5": (
            "bomb1,bomb2,bomb3,bomb4,bomb5",
            (
                "            sam  > orientation=UP color=BLUE speed=0.25 img=oryx/bullet1 shrinkfactor=2.0\n"
                + chopper_multishot_bomb_base
                + "            bomb1 > orientation=-0.7071,0.7071 color=RED speed=0.75 img=newset/bomb shrinkfactor=0.5\n"
                "            bomb2 > orientation=-0.3827,0.9239 color=RED speed=0.75 img=newset/bomb shrinkfactor=0.5\n"
                "            bomb3 > orientation=DOWN color=RED speed=0.75 img=newset/bomb shrinkfactor=0.5\n"
                "            bomb4 > orientation=0.3827,0.9239 color=RED speed=0.75 img=newset/bomb shrinkfactor=0.5\n"
                "            bomb5 > orientation=0.7071,0.7071 color=RED speed=0.75 img=newset/bomb shrinkfactor=0.5\n"
            ),
            ["bomb1", "bomb2", "bomb3", "bomb4", "bomb5"],
        ),
    }
    for tag, (stype, missile_lines, bombs) in chopper_multishot_specs.items():
        bomb_list = " ".join(bombs)
        write_chopper(
            tag,
            avatar_line=chopper_avatar(stype=stype, spread=chopper_multishot_spread),
            missile_lines=missile_lines,
            tank_bomb_lines="\n".join(
                f"        tank {b} > killSprite scoreChange=1" for b in bombs
            ),
            bomb_eos_line="        missile EOS > killSprite",
            bomb_sam_lines="\n".join(f"        {b} sam > killBoth" for b in bombs) + "\n",
            satellite_sam_line=f"satellite sam {bomb_list} > killBoth scoreChange=-1",
            cloud_sam_line=f"cloud sam {bomb_list} > killBoth",
        )

    write_chopper(
        "pierce_shot",
        avatar_line=avatar_default,
        missile_lines=missile_default,
        tank_bomb_lines=tank_kill,
        bomb_eos_line="        bomb wall EOS > killSprite",
    )

    for tag, spd in [("quick_dash_2tile", "2"), ("quick_dash_3tile", "3"), ("quick_dash_5tile", "5")]:
        write_chopper(
            tag,
            avatar_line=chopper_avatar(speed=spd),
            missile_lines=missile_default,
            tank_bomb_lines=tank_kill,
            bomb_eos_line=bomb_eos,
        )

    write_chopper(
        "shield_reflect",
        avatar_line=avatar_default,
        missile_lines=missile_default,
        tank_bomb_lines=tank_kill,
        bomb_eos_line=bomb_eos,
        avatar_sam_lines="        avatar sam > stepBack\n        sam avatar > reverseDirection",
    )

    write_chopper(
        "big_explosion_3rad",
        avatar_line=avatar_default,
        missile_lines=missile_default,
        extra_sprites=EXPLOSION_SPRITE,
        tank_bomb_lines=chopper_tank_bomb("bomb", "transformTo stype=explosion killSecond=True scoreChange=1"),
        bomb_eos_line="        bomb wall EOS > killSprite",
        extra_interactions="        tank explosion > killSprite\n        missile explosion > killSprite\n",
    )


def waves_avatar(speed: str = "1.0", stype: str = "sam", spread: str = "") -> str:
    spread_part = f" {spread}" if spread else ""
    return (
        f"        avatar  > ShootAvatar color=YELLOW stype={stype} speed={speed}"
        f"{spread_part} img=oryx/spaceship3 rotateInPlace=False"
    )


def waves_missile_sam(sf: str = "0.5") -> str:
    return (
        "            rock > orientation=LEFT speed=0.95 color=BLUE img=oryx/orb3\n"
        f"            sam  > orientation=RIGHT color=BLUE speed=1.0 img=oryx/orb1 shrinkfactor={sf}\n"
        "            laser > orientation=LEFT speed=0.3 color=RED shrinkfactor=0.75 img=newset/laser2_1"
    )


def waves_sams(n: int, sf: str = "0.5") -> str:
    head = "            rock > orientation=LEFT speed=0.95 color=BLUE img=oryx/orb3\n"
    sams = "\n".join(
        f"            sam{i} > orientation=RIGHT color=BLUE speed=1.0 img=oryx/orb1 shrinkfactor={sf}"
        for i in range(1, n + 1)
    )
    return head + sams + "\n            laser > orientation=LEFT speed=0.3 color=RED shrinkfactor=0.75 img=newset/laser2_1"


def gen_waves() -> None:
    default_avatar = waves_avatar()
    default_missile = waves_missile_sam()
    kill_both = "        alien sam > killBoth scoreChange=2"

    for tag, sf in [("big_shot_1x", "0.5"), ("big_shot_2x", "1.0"), ("big_shot_4x", "2.0")]:
        write_waves(tag, avatar_line=default_avatar, missile_lines=waves_missile_sam(sf), alien_sam_line=kill_both)

    for tag, n, spread in [
        ("multishot_2", 2, "alignShotToOrientation=False fireAllWeapons=True spreadPixels=0 spreadDegrees=45"),
        ("multishot_3", 3, "alignShotToOrientation=False fireAllWeapons=True spreadPixels=8"),
        ("multishot_5", 5, "alignShotToOrientation=False fireAllWeapons=True spreadPixels=0 spreadDegrees=22.5"),
    ]:
        st = " ".join(f"sam{i}" for i in range(1, n + 1))
        write_waves(
            tag,
            avatar_line=waves_avatar(stype=",".join(f"sam{i}" for i in range(1, n + 1)), spread=spread),
            missile_lines=waves_sams(n),
            alien_sam_line=f"        alien {st} > killBoth scoreChange=2",
            sam_types=st,
        )

    write_waves(
        "pierce_shot",
        avatar_line=default_avatar,
        missile_lines=default_missile,
        alien_sam_line="        alien sam > killSprite scoreChange=2",
    )

    for tag, spd in [("quick_dash_2tile", "2"), ("quick_dash_3tile", "3"), ("quick_dash_5tile", "5")]:
        write_waves(
            tag,
            avatar_line=waves_avatar(speed=spd),
            missile_lines=default_missile,
            alien_sam_line=kill_both,
        )

    write_waves(
        "shield_reflect",
        avatar_line=default_avatar,
        missile_lines=default_missile,
        alien_sam_line=kill_both,
        avatar_laser_lines="        avatar laser > stepBack\n        laser avatar > reverseDirection",
    )

    write_waves(
        "big_explosion_3rad",
        avatar_line=default_avatar,
        missile_lines=default_missile,
        extra_sprites=EXPLOSION_SPRITE,
        alien_sam_line="        alien sam > transformTo stype=explosion killSecond=True scoreChange=2",
        extra_interactions="        alien explosion > killSprite\n        missile explosion > killSprite\n",
    )


def jaws_avatar(speed: str = "1.0", stype: str = "torpedo", spread: str = "") -> str:
    spread_part = f" {spread}" if spread else ""
    return f"            avatar  > ShootAvatar color=YELLOW  stype={stype} img=newset/submarine speed={speed}{spread_part}"


def jaws_torpedo(sf: str = "0.5") -> str:
    return f"            torpedo > Missile color=YELLOW shrinkfactor={sf} img=oryx/orb2"


def jaws_torpedos(n: int, sf: str = "0.5") -> str:
    return "\n".join(
        f"            torpedo{i} > Missile color=YELLOW shrinkfactor={sf} img=oryx/orb2"
        for i in range(1, n + 1)
    )


def gen_jaws() -> None:
    default_avatar = jaws_avatar()
    default_torpedo = jaws_torpedo()
    shell_hits = (
        "        whale torpedo > transformTo stype=shell scoreChange=1\n"
        "        piranha torpedo > transformTo stype=shell scoreChange=1"
    )

    for tag, sf in [("big_shot_1x", "0.5"), ("big_shot_2x", "1.0"), ("big_shot_4x", "2.0")]:
        write_jaws(
            tag,
            avatar_line=default_avatar,
            torpedo_lines=jaws_torpedo(sf),
            fish_torpedo_lines=shell_hits,
        )

    for tag, n, spread in [
        ("multishot_2", 2, "fireAllWeapons=True spreadPixels=0 spreadDegrees=45"),
        ("multishot_3", 3, "fireAllWeapons=True spreadPixels=8"),
        ("multishot_5", 5, "fireAllWeapons=True spreadPixels=0 spreadDegrees=22.5"),
    ]:
        st = " ".join(f"torpedo{i}" for i in range(1, n + 1))
        fish_lines = "\n".join(
            f"        whale {t} > transformTo stype=shell scoreChange=1\n"
            f"        piranha {t} > transformTo stype=shell scoreChange=1"
            for t in [f"torpedo{i}" for i in range(1, n + 1)]
        )
        write_jaws(
            tag,
            avatar_line=jaws_avatar(stype=",".join(f"torpedo{i}" for i in range(1, n + 1)), spread=spread),
            torpedo_lines=jaws_torpedos(n),
            torpedo_types=st,
            fish_torpedo_lines=fish_lines,
            torpedo_fish_line=f"        {st} fish > killSprite",
        )

    write_jaws(
        "pierce_shot",
        avatar_line=default_avatar,
        torpedo_lines=default_torpedo,
        fish_torpedo_lines=(
            "        whale torpedo > killSprite scoreChange=1\n"
            "        piranha torpedo > killSprite scoreChange=1\n"
            "        torpedo shark > killSprite"
        ),
        torpedo_fish_line="",
    )

    for tag, spd in [("quick_dash_2tile", "2"), ("quick_dash_3tile", "3"), ("quick_dash_5tile", "5")]:
        write_jaws(
            tag,
            avatar_line=jaws_avatar(speed=spd),
            torpedo_lines=default_torpedo,
            fish_torpedo_lines=shell_hits,
        )

    write_jaws(
        "shield_reflect",
        avatar_line=default_avatar,
        torpedo_lines=default_torpedo,
        fish_torpedo_lines=shell_hits,
        avatar_fish_lines=(
            "        avatar whale > stepBack\n"
            "        whale avatar > reverseDirection\n"
            "        avatar piranha > stepBack\n"
            "        piranha avatar > reverseDirection"
        ),
    )

    write_jaws(
        "big_explosion_3rad",
        avatar_line=default_avatar,
        torpedo_lines=default_torpedo,
        extra_sprites=EXPLOSION_SPRITE,
        fish_torpedo_lines=(
            "        whale torpedo > transformTo stype=shell scoreChange=1\n"
            "        piranha torpedo > transformTo stype=shell scoreChange=1\n"
            "        shark torpedo > transformTo stype=explosion killSecond=True scoreChange=1"
        ),
        extra_interactions="        shark explosion > killSprite\n        fish explosion > killSprite\n",
    )


def main() -> None:
    gen_chopper()
    gen_waves()
    gen_jaws()
    print("generated ikaruga-style variants for chopper, waves, jaws")


if __name__ == "__main__":
    main()
