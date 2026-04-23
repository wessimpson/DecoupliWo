#!/usr/bin/env python3
from __future__ import annotations

import html
import pathlib
import zipfile


OUT = pathlib.Path("slides/object_centric_world_model_deck.pptx")

EMU_W = 12_192_000
EMU_H = 6_858_000

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def emu(inches: float) -> int:
    return int(round(inches * 914_400))


def color_fill(hex_color: str) -> str:
    return f'<a:solidFill><a:srgbClr val="{hex_color}"/></a:solidFill>'


def text_paragraph(text: str, size: int = 1800, color: str = "172033", bold: bool = False, bullet: bool = False) -> str:
    bullet_xml = '<a:buChar char="&#8226;"/>' if bullet else "<a:buNone/>"
    bold_attr = ' b="1"' if bold else ""
    return (
        "<a:p>"
        f"{bullet_xml}"
        f'<a:r><a:rPr lang="en-US" sz="{size}"{bold_attr}>{color_fill(color)}</a:rPr>'
        f"<a:t>{esc(text)}</a:t></a:r>"
        "</a:p>"
    )


def textbox(
    shape_id: int,
    name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    paragraphs: list[str],
    fill: str = "FFFFFF",
    line: str = "D5DBE8",
    radius: str = "roundRect",
    margin_l: int = 150_000,
    margin_r: int = 150_000,
    margin_t: int = 90_000,
    margin_b: int = 90_000,
) -> str:
    return (
        "<p:sp>"
        "<p:nvSpPr>"
        f'<p:cNvPr id="{shape_id}" name="{esc(name)}"/>'
        "<p:cNvSpPr/><p:nvPr/>"
        "</p:nvSpPr>"
        "<p:spPr>"
        f'<a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
        f'<a:prstGeom prst="{radius}"><a:avLst/></a:prstGeom>'
        f"{color_fill(fill)}"
        f'<a:ln w="12700"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
        "</p:spPr>"
        f'<p:txBody><a:bodyPr wrap="square" lIns="{margin_l}" rIns="{margin_r}" tIns="{margin_t}" bIns="{margin_b}"/>'
        "<a:lstStyle/>"
        + "".join(paragraphs)
        + "</p:txBody>"
        "</p:sp>"
    )


def line(shape_id: int, x1: float, y1: float, x2: float, y2: float, color: str = "596579") -> str:
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    flip_h = ' flipH="1"' if x2 < x1 else ""
    flip_v = ' flipV="1"' if y2 < y1 else ""
    return (
        "<p:cxnSp>"
        "<p:nvCxnSpPr>"
        f'<p:cNvPr id="{shape_id}" name="connector {shape_id}"/>'
        "<p:cNvCxnSpPr/><p:nvPr/>"
        "</p:nvCxnSpPr>"
        "<p:spPr>"
        f'<a:xfrm{flip_h}{flip_v}><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(max(w, 0.001))}" cy="{emu(max(h, 0.001))}"/></a:xfrm>'
        '<a:prstGeom prst="line"><a:avLst/></a:prstGeom>'
        f'<a:ln w="19050"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:tailEnd type="triangle"/></a:ln>'
        "</p:spPr>"
        "</p:cxnSp>"
    )


def slide_xml(shapes: list[str], bg: str = "F7F9FC") -> str:
    return (
        f'<p:sld xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}" xmlns:p="{NS["p"]}">'
        "<p:cSld>"
        "<p:bg><p:bgPr>"
        f"{color_fill(bg)}"
        "<a:effectLst/>"
        "</p:bgPr></p:bg>"
        "<p:spTree>"
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        + "".join(shapes)
        + "</p:spTree>"
        "</p:cSld>"
        "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>"
        "</p:sld>"
    )


def title(shape_id: int, text: str, subtitle: str = "") -> str:
    paragraphs = [text_paragraph(text, size=2850, color="111827", bold=True)]
    if subtitle:
        paragraphs.append(text_paragraph(subtitle, size=1320, color="4B5563"))
    return textbox(shape_id, "title", 0.38, 0.22, 12.55, 0.72, paragraphs, fill="F7F9FC", line="F7F9FC", radius="rect")


def footer(shape_id: int, text: str) -> str:
    return textbox(shape_id, "footer", 0.55, 7.08, 12.2, 0.25, [text_paragraph(text, size=900, color="6B7280")], fill="F7F9FC", line="F7F9FC", radius="rect")


def slide_1() -> str:
    shapes: list[str] = [
        title(2, "Object-Centric Neural Game Engine", "Shared architecture from Soyuj's slot-GNN baseline and Pranit's optimized WM"),
        textbox(
            3,
            "key idea",
            0.55,
            1.25,
            4.7,
            4.35,
            [
                text_paragraph("Key Idea", size=2100, bold=True, color="0F172A"),
                text_paragraph("Learn structured dynamics, not pixels: S_t, a_t, rule -> S_{t+1}.", bullet=True),
                text_paragraph("Objects are fixed slots: x, y, vx, vy, size, type, mask/alive flag.", bullet=True),
                text_paragraph("Rules are explicit conditioning variables: normal, gravity, teleport.", bullet=True),
                text_paragraph("Same schema supports Pong and Breakout-lite: ball, paddle, blocks.", bullet=True),
                text_paragraph("Goal: swap rules and run counterfactual rollouts in the learned engine.", bullet=True),
            ],
            fill="EEF6FF",
            line="99BDEB",
        ),
        textbox(
            4,
            "envs",
            0.55,
            5.83,
            4.7,
            0.82,
            [
                text_paragraph("Environments", size=1450, bold=True, color="0F172A"),
                text_paragraph("Pong: normal / gravity / teleport. Breakout-lite: same object schema + blocks.", size=1200, color="334155"),
            ],
            fill="FFFFFF",
            line="CBD5E1",
        ),
    ]
    boxes = [
        ("Structured state API / RAM", "S_t object slots + action + rule id"),
        ("Graph builder", "hybrid edges: relative pos, relative vel, object type"),
        ("Rule/action-conditioned MPNN", "node encoder -> edge messages -> aggregate -> node update"),
        ("Dynamics heads", "slot update, mask/alive, event logits"),
        ("Neural rollout engine", "S_{t+1}; repeat for counterfactual simulation"),
    ]
    x = 6.05
    y = 1.12
    for idx, (head, body) in enumerate(boxes):
        sid = 10 + idx
        shapes.append(
            textbox(
                sid,
                head,
                x,
                y + idx * 1.02,
                6.45,
                0.72,
                [text_paragraph(head, size=1450, bold=True), text_paragraph(body, size=1050, color="475569")],
                fill="FFFFFF" if idx % 2 == 0 else "F8FAFC",
                line="CBD5E1",
            )
        )
        if idx < len(boxes) - 1:
            shapes.append(line(30 + idx, x + 3.23, y + idx * 1.02 + 0.74, x + 3.23, y + idx * 1.02 + 1.02))
    shapes.append(
        textbox(
            20,
            "architecture label",
            6.05,
            6.35,
            6.45,
            0.38,
            [text_paragraph("Right side: complete no-pixel object-GNN pipeline", size=1250, bold=True, color="1D4ED8")],
            fill="DBEAFE",
            line="93C5FD",
        )
    )
    shapes.append(footer(40, "Object-centric interface: no CNN / VAE / image reconstruction; all learning happens over structured object state."))
    return slide_xml(shapes)


def slide_2() -> str:
    shapes: list[str] = [
        title(2, "Soyuj's World Model & Results", "Baseline contribution: playable state-based editable world model with object slots"),
        textbox(
            3,
            "soyuj established",
            0.55,
            1.14,
            5.15,
            4.85,
            [
                text_paragraph("What Soyuj Established", size=2050, bold=True),
                text_paragraph("Custom Pong variants and Breakout-lite environments.", bullet=True),
                text_paragraph("Counterfactual data collection: same state/action stepped under each rule.", bullet=True),
                text_paragraph("Shared object-slot format across games: ball, paddle, blocks.", bullet=True),
                text_paragraph("Rule-conditioned slot GNN: type embeddings, rule embeddings, action injected into paddle nodes.", bullet=True),
                text_paragraph("Playable learned world model through pygame.", bullet=True),
            ],
            fill="F0FDF4",
            line="86EFAC",
        ),
        textbox(
            4,
            "baseline architecture",
            6.05,
            1.15,
            6.55,
            2.02,
            [
                text_paragraph("Baseline Architecture", size=1850, bold=True),
                text_paragraph("slots -> node encoder -> fully connected message passing -> residual slot decoder", bullet=True),
                text_paragraph("Loss: one-step slot MSE + contrastive latent loss + mask/alive loss", bullet=True),
                text_paragraph("Rule transfer support: explicit rule ids and rule ablation evaluation", bullet=True),
            ],
            fill="FFFFFF",
            line="CBD5E1",
        ),
        textbox(
            5,
            "results",
            6.05,
            3.45,
            6.55,
            2.35,
            [
                text_paragraph("Observed Result Pattern", size=1850, bold=True),
                text_paragraph("Prior playable run reached best val_slot_rmse ~= 10.0.", bullet=True),
                text_paragraph("Gameplay still exposed instability: ball drift/jitter, wrong bounces, weak long rollouts.", bullet=True),
                text_paragraph("Collision/rare mechanics were much worse than ordinary free motion.", bullet=True),
                text_paragraph("Lesson: low one-step error is not enough for a reliable neural game engine.", bullet=True),
            ],
            fill="FFF7ED",
            line="FDBA74",
        ),
        textbox(
            6,
            "metric bar",
            0.55,
            6.22,
            12.05,
            0.56,
            [
                text_paragraph(
                    "Baseline takeaway: Soyuj built the editable object-WM pipeline; Pranit's work targets rollout stability, events, and rule separation.",
                    size=1280,
                    bold=True,
                    color="7C2D12",
                )
            ],
            fill="FFEDD5",
            line="FDBA74",
        ),
    ]
    shapes.append(footer(40, "Numbers shown are local run metrics; gameplay quality was judged from learned-engine playbacks, not just one-step validation."))
    return slide_xml(shapes)


def slide_3() -> str:
    shapes: list[str] = [
        title(2, "Areas of Optimization Addressed in Pranit's WM", "Turning a slot predictor into a more stable neural simulator"),
        textbox(
            3,
            "problems",
            0.55,
            1.12,
            5.25,
            4.9,
            [
                text_paragraph("Failure Modes Observed", size=2050, bold=True),
                text_paragraph("Ball dynamics unstable: random direction and speed changes.", bullet=True),
                text_paragraph("Paddle drifted because actuator physics were learned freely.", bullet=True),
                text_paragraph("Collisions and wall bounces underfit relative to smooth motion.", bullet=True),
                text_paragraph("One-step loss dominated; autoregressive rollouts compounded errors.", bullet=True),
                text_paragraph("Rule effects could be entangled instead of cleanly separated.", bullet=True),
                text_paragraph("Teleport/wrap remains a special discontinuous-event challenge.", bullet=True),
            ],
            fill="FEF2F2",
            line="FCA5A5",
        ),
        textbox(
            4,
            "optimizations",
            6.05,
            1.12,
            6.55,
            4.9,
            [
                text_paragraph("Optimizations Added", size=2050, bold=True),
                text_paragraph("Bounded residual deltas and ball velocity clamp for rollout stability.", bullet=True),
                text_paragraph("Deterministic paddle kinematics from action, speed, and dt.", bullet=True),
                text_paragraph("Feature-weighted slot loss + delta loss + free-flight kinematic loss.", bullet=True),
                text_paragraph("Multi-step rollout loss, rare-event sampler, event-weighted supervision.", bullet=True),
                text_paragraph("Counterfactual rule loss: same state/action, all rule targets.", bullet=True),
                text_paragraph("Event-specific metrics expose paddle hits, wall bounces, and wraps separately.", bullet=True),
            ],
            fill="EFF6FF",
            line="93C5FD",
        ),
        textbox(
            5,
            "next",
            1.1,
            6.28,
            10.9,
            0.52,
            [
                text_paragraph(
                    "Next step: velocity-integrated decoder + event/rule-aware jump head for teleport and collision corrections.",
                    size=1280,
                    bold=True,
                    color="1E3A8A",
                )
            ],
            fill="DBEAFE",
            line="60A5FA",
        ),
    ]
    shapes.append(footer(40, "Pranit WM optimization target: better physical consistency, long-horizon rollout stability, and rule-transfer behavior."))
    return slide_xml(shapes)


def rels_xml(rels: list[tuple[str, str, str]]) -> str:
    body = "".join(
        f'<Relationship Id="{rid}" Type="{typ}" Target="{esc(target)}"/>'
        for rid, typ, target in rels
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + body
        + "</Relationships>"
    )


def content_types() -> str:
    overrides = [
        ("/ppt/presentation.xml", "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"),
        ("/ppt/slideMasters/slideMaster1.xml", "application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"),
        ("/ppt/slideLayouts/slideLayout1.xml", "application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"),
        ("/ppt/theme/theme1.xml", "application/vnd.openxmlformats-officedocument.theme+xml"),
        ("/ppt/presProps.xml", "application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"),
        ("/ppt/viewProps.xml", "application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"),
        ("/ppt/tableStyles.xml", "application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"),
    ]
    for idx in range(1, 4):
        overrides.append((f"/ppt/slides/slide{idx}.xml", "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"))
    override_xml = "".join(f'<Override PartName="{name}" ContentType="{ctype}"/>' for name, ctype in overrides)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + override_xml
        + "</Types>"
    )


def presentation_xml() -> str:
    return (
        f'<p:presentation xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}" xmlns:p="{NS["p"]}" saveSubsetFonts="1">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        '<p:sldIdLst>'
        '<p:sldId id="256" r:id="rId2"/>'
        '<p:sldId id="257" r:id="rId3"/>'
        '<p:sldId id="258" r:id="rId4"/>'
        '</p:sldIdLst>'
        f'<p:sldSz cx="{EMU_W}" cy="{EMU_H}" type="wide"/>'
        '<p:notesSz cx="6858000" cy="9144000"/>'
        '<p:defaultTextStyle/>'
        '</p:presentation>'
    )


def slide_master_xml() -> str:
    return (
        f'<p:sldMaster xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}" xmlns:p="{NS["p"]}">'
        '<p:cSld><p:spTree>'
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        '</p:spTree></p:cSld>'
        '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
        '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
        '<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>'
        '</p:sldMaster>'
    )


def slide_layout_xml() -> str:
    return (
        f'<p:sldLayout xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}" xmlns:p="{NS["p"]}" type="blank" preserve="1">'
        '<p:cSld name="Blank"><p:spTree>'
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        '</p:spTree></p:cSld>'
        '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>'
        '</p:sldLayout>'
    )


def theme_xml() -> str:
    return (
        f'<a:theme xmlns:a="{NS["a"]}" name="ObjectCentric">'
        '<a:themeElements>'
        '<a:clrScheme name="ObjectCentric">'
        '<a:dk1><a:srgbClr val="111827"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>'
        '<a:dk2><a:srgbClr val="1F2937"/></a:dk2><a:lt2><a:srgbClr val="F7F9FC"/></a:lt2>'
        '<a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="16A34A"/></a:accent2>'
        '<a:accent3><a:srgbClr val="F97316"/></a:accent3><a:accent4><a:srgbClr val="DC2626"/></a:accent4>'
        '<a:accent5><a:srgbClr val="7C3AED"/></a:accent5><a:accent6><a:srgbClr val="0891B2"/></a:accent6>'
        '<a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink>'
        '</a:clrScheme>'
        '<a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>'
        '<a:fmtScheme name="Clean"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>'
        '<a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>'
        '<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>'
        '<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>'
        '</a:themeElements></a:theme>'
    )


def write_deck() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    slides = [slide_1(), slide_2(), slide_3()]
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types())
        zf.writestr("_rels/.rels", rels_xml([("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "ppt/presentation.xml")]))
        zf.writestr(
            "ppt/_rels/presentation.xml.rels",
            rels_xml(
                [
                    ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "slideMasters/slideMaster1.xml"),
                    ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", "slides/slide1.xml"),
                    ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", "slides/slide2.xml"),
                    ("rId4", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", "slides/slide3.xml"),
                    ("rId5", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps", "presProps.xml"),
                    ("rId6", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps", "viewProps.xml"),
                    ("rId7", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles", "tableStyles.xml"),
                    ("rId8", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "theme/theme1.xml"),
                ]
            ),
        )
        zf.writestr("ppt/presentation.xml", presentation_xml())
        zf.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml())
        zf.writestr(
            "ppt/slideMasters/_rels/slideMaster1.xml.rels",
            rels_xml(
                [
                    ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml"),
                    ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "../theme/theme1.xml"),
                ]
            ),
        )
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml())
        zf.writestr(
            "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
            rels_xml([("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "../slideMasters/slideMaster1.xml")]),
        )
        zf.writestr("ppt/theme/theme1.xml", theme_xml())
        zf.writestr("ppt/presProps.xml", f'<p:presentationPr xmlns:p="{NS["p"]}"/>')
        zf.writestr("ppt/viewProps.xml", f'<p:viewPr xmlns:p="{NS["p"]}"/>')
        zf.writestr("ppt/tableStyles.xml", f'<a:tblStyleLst xmlns:a="{NS["a"]}" def="{{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}}"/>')
        for idx, slide in enumerate(slides, start=1):
            zf.writestr(f"ppt/slides/slide{idx}.xml", slide)
            zf.writestr(
                f"ppt/slides/_rels/slide{idx}.xml.rels",
                rels_xml([("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml")]),
            )


if __name__ == "__main__":
    write_deck()
    print(OUT.resolve())
