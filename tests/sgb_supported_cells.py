"""Supported Geometry cells 1-12: the flip-chip Disk Qubit's unit cells as SGB-native scenes.

Dimensions are those of the OrcaPen SC PDK Disk Qubit drawn with ``curve_sampling="simulation"``: 32-gons above
100 um diameter, 16-gons below 20 um.  The geometry is drawn clean: every union is computed once from exact line
intersections, so no boolean leaves nanometre steps, slits, cracks or split vertices.  One function per unit
cell returns a :class:`Scene`; the combinations the Disk Qubit really has are their own functions.

In cell 10 only the lumped sheet and the leads are drawn.  The 5 x 0.09 um cross the PDK draws on layer 22/0 is
the fabrication junction; it is not part of the simulation layer stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sgb_supported_kit import (
    Conductor,
    Expectation,
    Point,
    Port,
    Scene,
    Um,
    chord,
    conductor_shape,
    die_box,
    die_footprint,
    fillet,
    ground,
    intersect,
    lattice,
    ngon,
    polar,
    quarter_turn,
    rect,
    rotate,
    round_cap,
    to_lattice,
)
from shapely.geometry import box
from shapely.ops import unary_union

from scgsim.geometry.sgb.records.grid import exact_z
from scgsim.geometry.sgb.records.prisms import PrismRecord

DISK_R = 110.5
DISK_R_TEST_BUILD = (
    119.83949086316494  # include_ground=False, junction_lead_span=0, floating claws
)
CLEARANCE_R = 350.0
HANDOFF = 412.3141121612922  # where the claw CPWs and the 45-degree lines leave the clearance region
CPW_W, CPW_G = 3.0, 40.0
CLAW_GAP, CLAW_WIDTH, CLAW_SPAN = 60.0, 60.0, 45.0
CORNER_R = 5.0
RO_W, RO_G, XY_W, XY_G = 7.0, 6.0, 4.0, 6.0
XY_TIP_R = 412.0
JUNCTION_ANGLE = -45.0
DISK_TERMINAL = (
    233.5,
    0.0,
)  # junction frame: the junction axis on +x, rotated by JUNCTION_ANGLE into the layout
GROUND_TERMINAL = (237.5, -4.0)
LEAD_W = 2.0
LEAD_SPAN = 20.0  # degrees of the disk arc the radial lead fan lands on
GROUND_LEAD_END_R = 360.0
SHEET_W = 0.27
CIRCLE = 32  # vertices of every polygon circle above 100 um diameter
GROUND_HALF = 500.0  # the 1000 um die


def clearance_vertex(k: int) -> Um:
    """Vertex k of the clearance 32-gon, at k * 11.25 degrees."""
    return polar(CLEARANCE_R, 360.0 * k / CIRCLE)


def cell01_disk(r: float = DISK_R) -> Scene:
    """Cell 1: the disk island, a 32-gon, alone on a die (SA all round it)."""
    return Scene(
        f"cell01_disk_r{r:.5g}",
        (1,),
        die_box(r + 20),
        [Conductor("disk", "disk", to_lattice(ngon(r, CIRCLE)))],
    )


def cell02_clearance() -> Scene:
    """Cells 2 and 12: the 1000 um ground square with the 32-gon clearance hole."""
    die = die_box(GROUND_HALF)
    return Scene(
        "cell02_clearance",
        (2, 12),
        die,
        [ground("ground", die, [to_lattice(ngon(CLEARANCE_R, CIRCLE))])],
    )


def disk_in_clearance() -> Scene:
    """Cells 1 + 2 + 12: the disk in the clearance of the ground."""
    s = cell02_clearance()
    disk = Conductor("disk", "disk", to_lattice(ngon(DISK_R, CIRCLE)))
    return Scene("disk_in_clearance", (1, 2, 12), s.die, [disk, *s.conductors])


def cell06_cpw_straight(length: float = 100.0, ground_w: float = 50.0) -> Scene:
    """Cell 6: a Manhattan CPW straight across the die (its ends and the grounds' outer edges on the die edge)."""
    y = CPW_W / 2 + CPW_G + ground_w
    return Scene(
        "cell06_cpw_straight",
        (6,),
        (0.0, -y, length, y),
        [
            Conductor(
                "centre", "centre", to_lattice(rect(0, -CPW_W / 2, length, CPW_W / 2))
            ),
            Conductor(
                "ground_lo",
                "Ground",
                to_lattice(rect(0, -y, length, -CPW_W / 2 - CPW_G)),
            ),
            Conductor(
                "ground_hi", "Ground", to_lattice(rect(0, CPW_W / 2 + CPW_G, length, y))
            ),
        ],
    )


def cell07_cpw_45(w: float, g: float, half: float = 60.0) -> Scene:
    """Cell 7: a 45-degree CPW straight across a square die, corner to corner (lines y = x + c, c on the lattice)."""

    def above(c: float) -> list[Um]:  # the part of the die above the line y = x + c
        return [(-half, -half + c), (half - c, half), (-half, half)]

    h, s = (
        lattice(v) * 0.001 for v in (w / 2 * math.sqrt(2), (w / 2 + g) * math.sqrt(2))
    )
    centre = [
        (-half + h, -half),
        (half, half - h),
        (half, half),
        (half - h, half),
        (-half, -half + h),
        (-half, -half),
    ]
    return Scene(
        f"cell07_cpw45_w{w:g}",
        (7,),
        die_box(half),
        [
            Conductor("centre", "centre", to_lattice(centre)),
            Conductor("ground_hi", "Ground", to_lattice(above(s))),
            Conductor("ground_lo", "Ground", to_lattice([(y, x) for x, y in above(s)])),
        ],
    )


def _ro_parts() -> tuple[list[Um], list[Um]]:
    """Cells 7 + 8 (readout): the 135-degree line with its round tip at the centre and the slot it cuts.

    The metal cap has radius 3.5 with its tip at r = 0.  The slot replaces the 135-degree vertex of the clearance
    and meets the two chords beside it.
    """
    axis, h, hs = 135.0, RO_W / 2, RO_W / 2 + RO_G
    end, mask_end = HANDOFF * math.sqrt(2), HANDOFF * math.sqrt(2) + RO_G
    lead = [rotate(p, axis) for p in [(end, -h), (end, h)]]
    metal = [*round_cap(polar(h, axis), h, axis + 180, inscribed_steps=True), *lead]
    lo = [rotate(p, axis) for p in [(0, -hs), (mask_end, -hs)]]
    hi = [rotate(p, axis) for p in [(mask_end, hs), (0, hs)]]
    k = 12  # the 135-degree vertex of the 32-gon
    p_lo = intersect(
        lo[0], lo[1], *chord(CLEARANCE_R, 360 * (k - 1) / CIRCLE, 360 * k / CIRCLE)
    )
    p_hi = intersect(
        hi[0], hi[1], *chord(CLEARANCE_R, 360 * k / CIRCLE, 360 * (k + 1) / CIRCLE)
    )
    return metal, [p_lo, lo[1], hi[0], p_hi]


def _xy_parts() -> tuple[list[Um], list[Um]]:
    """Cells 7 + 8 (drive): the 225-degree line, metal cap r 2 at r 412 inside a mask cap r 8, both 16-gons."""
    axis, h, hs = 225.0, XY_W / 2, XY_W / 2 + XY_G
    cap_centre = XY_TIP_R + h
    end, mask_end = HANDOFF * math.sqrt(2), HANDOFF * math.sqrt(2) + XY_G

    def line(half_w: float, stop: float) -> list[Um]:
        local = [
            *round_cap((cap_centre, 0.0), half_w, 180.0, inscribed_steps=True),
            (stop, -half_w),
            (stop, half_w),
        ]
        return [rotate(p, axis) for p in local]

    return line(h, end), line(hs, mask_end)


def _clearance_with(
    replace: dict[int, list[Um]], after: dict[int, list[Um]] | None = None
) -> list[Um]:
    """The clearance 32-gon in increasing angle, vertex k replaced by ``replace[k]``, ``after[k]`` inserted after it."""
    ring: list[Um] = []
    for k in range(CIRCLE):
        ring += replace.get(k, [clearance_vertex(k)])
        ring += (after or {}).get(k, [])
    return ring


def ro_through_clearance() -> Scene:
    """Cells 2 + 7 + 8 + 12: the readout line from its round tip out through the clearance at its 135-degree vertex.

    The slot ends one gap past the line end, inside the ground.
    """
    metal, slot = _ro_parts()
    die = die_box(GROUND_HALF)
    hole = to_lattice(_clearance_with({12: slot}))
    return Scene(
        "ro_through_clearance",
        (2, 7, 8, 12),
        die,
        [Conductor("ro", "ro", to_lattice(metal)), ground("ground", die, [hole])],
    )


def xy_tip() -> Scene:
    """Cells 7 + 8 + 12: the drive line ending in a round tip inside the ground, metal cap in a mask cap."""
    metal, mask = _xy_parts()
    die = (-480.0, -480.0, -260.0, -260.0)
    return Scene(
        "xy_tip",
        (7, 8, 12),
        die,
        [
            Conductor("xy", "xy", to_lattice(metal)),
            ground("ground", die, [to_lattice(mask)]),
        ],
    )


def d0_die() -> Scene:
    """The control die, clean: the clearance with the readout slot, the drive slot, both lines, the ground square."""
    ro, slot = _ro_parts()
    xy, mask = _xy_parts()
    die = die_box(GROUND_HALF)
    holes = [to_lattice(_clearance_with({12: slot})), to_lattice(mask)]
    return Scene(
        "d0_die",
        (2, 7, 8, 12),
        die,
        [
            Conductor("ro", "ro", to_lattice(ro)),
            Conductor("xy", "xy", to_lattice(xy)),
            ground("ground", die, holes),
        ],
    )


@dataclass
class Claw:
    """One claw on local +x (cells 3, 4, 5, 6): its metal outline and the mask that replaces the clearance's
    0-degree vertex in the ground hole (increasing angle), with or without the grounded tail."""

    metal: list[Um]
    mask: list[Um]


def claw(disk_r: float = DISK_R, grounded: bool = False) -> Claw:
    """The claw arc (cell 3) with rounded inner corners (cell 4), its taper (cell 5) and CPW (cell 6)."""
    r_in = disk_r + CLAW_GAP
    r_out = r_in + CLAW_WIDTH
    hw = CPW_W / 2
    up, lo = (
        polar(r_out, CLAW_SPAN / 2),
        polar(r_out, -CLAW_SPAN / 2),
    )  # outer corners: the taper lands here
    neck_up, neck_lo = (CLEARANCE_R, hw), (CLEARANCE_R, -hw)
    inner = [
        polar(r_in, -CLAW_SPAN / 2 + 11.25 * k) for k in range(5)
    ]  # 4 chords of 11.25 degrees
    arc = [
        *fillet(lo, inner[0], inner[1], CORNER_R, 4),
        *inner[1:4],
        *fillet(inner[3], inner[4], up, CORNER_R, 4),
    ]  # from the lower radial edge round to the upper one (increasing angle)
    metal = [(HANDOFF, -hw), (HANDOFF, hw), neck_up, up, *arc[::-1], lo, neck_lo]
    # The mask outside the clearance: the CPW gap strip to HANDOFF + gap, and the taper edge offset by the gap.
    mask_end, mask_half = HANDOFF + CPW_G, hw + CPW_G
    side = (up[0] - neck_up[0], up[1] - neck_up[1])
    length = math.hypot(*side)
    off = (CPW_G * side[1] / length, -CPW_G * side[0] / length)
    o_up = (
        (neck_up[0] + off[0], neck_up[1] + off[1]),
        (up[0] + off[0], up[1] + off[1]),
    )
    o_lo = ((o_up[0][0], -o_up[0][1]), (o_up[1][0], -o_up[1][1]))
    c_up = intersect(*o_up, *chord(CLEARANCE_R, 0.0, 11.25))
    c_lo = intersect(*o_lo, *chord(CLEARANCE_R, -11.25, 0.0))
    s_up = intersect(*o_up, (0.0, mask_half), (1.0, mask_half))
    s_lo = (s_up[0], -s_up[1])
    if grounded:  # the tail joins the CPW to the ground past the mask end: the claw becomes a peninsula
        peninsula = [(mask_end, -hw), neck_lo, lo, *arc, up, neck_up, (mask_end, hw)]
        mask = [
            c_lo,
            s_lo,
            (mask_end, -mask_half),
            *peninsula,
            (mask_end, mask_half),
            s_up,
            c_up,
        ]
    else:
        mask = [c_lo, s_lo, (mask_end, -mask_half), (mask_end, mask_half), s_up, c_up]
    return Claw(metal, mask)


def claw_scene(disk_r: float = DISK_R, grounded: bool = False) -> Scene:
    """Cells 1-6 + 12: one claw (arc, rounded inner corners, taper, CPW) on +x with its mask, the disk, the ground."""
    c = claw(disk_r, grounded)
    hole = to_lattice(c.mask) + to_lattice(
        [clearance_vertex(k) for k in range(1, CIRCLE)]
    )
    conductors = [Conductor("disk", "disk", to_lattice(ngon(disk_r, CIRCLE)))]
    if not grounded:
        conductors.append(Conductor("claw", "claw", to_lattice(c.metal)))
    conductors.append(ground("ground", die_box(GROUND_HALF), [hole]))
    name = f"claw_{'grounded' if grounded else 'floating'}_r{disk_r:.5g}"
    return Scene(name, (1, 2, 3, 4, 5, 6, 12), die_box(GROUND_HALF), conductors)


def _junction(points: list[Um]) -> list[Um]:
    return [rotate(p, JUNCTION_ANGLE) for p in points]


def _disk_with_lead(disk_r: float, fan: bool) -> list[Um]:
    """Cells 1 + 9 + 10, junction frame: the disk 32-gon with its 2 um lead and round cap r 1 on the contact.

    With ``fan`` the radial taper widens from the 2 um neck to a 20-degree arc of the disk, landing on its chords.
    """
    d = DISK_TERMINAL
    lo, up = (d[0], d[1] - LEAD_W / 2), (d[0], d[1] + LEAD_W / 2)
    if fan:
        far_lo, far_up = polar(disk_r, -LEAD_SPAN / 2), polar(disk_r, LEAD_SPAN / 2)
    else:
        far_lo, far_up = (0.0, lo[1]), (0.0, up[1])
    x_lo = intersect(lo, far_lo, *chord(disk_r, -11.25, 0.0))
    x_up = intersect(up, far_up, *chord(disk_r, 0.0, 11.25))
    cap = round_cap(d, LEAD_W / 2, 0.0, inscribed_steps=False)
    ring = [
        polar(disk_r, 360.0 * k / CIRCLE) for k in range(-31, 0)
    ]  # 11.25 ... 348.75, increasing angle
    return [*ring, x_lo, *cap, x_up]


def _tongue(fan: bool) -> list[Um]:
    """Cells 9 + 10, junction frame: the ground lead (2 um, cap r 1 on the contact) as a tongue of the ground.

    With ``fan`` it widens to a 20-degree arc at r 360 inside the ground.  Increasing angle in the hole.
    """
    g = GROUND_TERMINAL
    lo, up = (g[0], g[1] - LEAD_W / 2), (g[0], g[1] + LEAD_W / 2)
    if fan:
        far_lo, far_up = (
            polar(GROUND_LEAD_END_R, -LEAD_SPAN / 2),
            polar(GROUND_LEAD_END_R, LEAD_SPAN / 2),
        )
        x_lo = intersect(lo, far_lo, *chord(CLEARANCE_R, -11.25, 0.0))
        x_up = intersect(up, far_up, *chord(CLEARANCE_R, 0.0, 11.25))
    else:
        x_lo = intersect(lo, (1.0, lo[1]), *chord(CLEARANCE_R, -11.25, 0.0))
        x_up = intersect(up, (1.0, up[1]), *chord(CLEARANCE_R, -11.25, 0.0))
    cap = round_cap(g, LEAD_W / 2, 180.0, inscribed_steps=False)[
        ::-1
    ]  # 270 -> 180 -> 90 degrees
    return [x_lo, *cap, x_up]


def _sheet() -> list[Point]:
    """Cell 10: the lumped junction sheet, axis-aligned in the layout, from one cap centre to the other."""
    d, g = (
        rotate(DISK_TERMINAL, JUNCTION_ANGLE),
        rotate(GROUND_TERMINAL, JUNCTION_ANGLE),
    )
    return to_lattice(rect(d[0] - SHEET_W / 2, g[1], d[0] + SHEET_W / 2, d[1]))


def junction_scene(fan: bool, port: bool = True) -> Scene:
    """Cells 1 + 2 + 9 + 10 + 12: the disk with its lead (and fan), the ground tongue (and fan), the lumped sheet."""
    disk_r = DISK_R if fan else DISK_R_TEST_BUILD
    k_vertex = 28  # the 315-degree (junction-axis) vertex of the clearance
    tongue = _junction(_tongue(fan))
    hole = (
        _clearance_with({k_vertex: tongue})
        if fan
        else _clearance_with({}, {k_vertex - 1: tongue})
    )
    name = f"junction_{'fan' if fan else 'straight'}{'' if port else '_noport'}"
    s = Scene(
        name,
        (1, 2, 9, 10, 12) if fan else (1, 2, 10, 12),
        die_box(GROUND_HALF),
        [
            Conductor(
                "disk", "disk", to_lattice(_junction(_disk_with_lead(disk_r, fan)))
            ),
            ground("ground", die_box(GROUND_HALF), [to_lattice(hole)]),
        ],
    )
    if port:
        s.ports.append(Port("junction", _sheet(), ("disk", "ground")))
    return s


def d1_die(grounded: bool) -> Scene:
    """The qubit die, clean: the defaults (grounded claws, fans) or the test build (floating claws, straight leads)."""
    disk_r = DISK_R if grounded else DISK_R_TEST_BUILD
    c = claw(disk_r, grounded)
    tongue = to_lattice(_junction(_tongue(grounded)))
    hole: list[Point] = []
    for k in range(CIRCLE):
        if k % 8 == 0:
            hole += quarter_turn(to_lattice(c.mask), k // 8)
        elif k == 28 and grounded:
            hole += tongue
        else:
            hole += to_lattice([clearance_vertex(k)])
            if k == 27 and not grounded:
                hole += tongue
    conductors = [
        Conductor(
            "disk", "disk", to_lattice(_junction(_disk_with_lead(disk_r, grounded)))
        )
    ]
    if not grounded:
        conductors += [
            Conductor(f"claw_{n}", f"claw_{n}", quarter_turn(to_lattice(c.metal), q))
            for q, n in enumerate("ENWS")
        ]
    conductors.append(ground("ground", die_box(GROUND_HALF), [hole]))
    return Scene(
        f"d1_die_{'defaults' if grounded else 'test_build'}",
        (*range(1, 11), 12),
        die_box(GROUND_HALF),
        conductors,
        [Port("junction", _sheet(), ("disk", "ground"))],
    )


# --------------------------------------------------------------------------- cell 11: the flip chip

D0_METAL = ("0", "0.1")
BUMP_SPAN = ("0.1", "8.1")
D1_METAL = ("8.1", "8.2")


def cell11_bumps() -> Scene:
    """Cell 11, flip chip: both grounds with the clearance, three 20 um bumps on r 400 joining them.

    The control die's ground lies on z 0..0.1, the qubit die's on 8.1..8.2 under the upper substrate; the bumps
    stand at 11.25, 22.5 and 33.75 degrees between them.
    """
    die = die_box(GROUND_HALF)
    hole = [to_lattice(ngon(CLEARANCE_R, CIRCLE))]
    bumps = []
    for k in (1, 2, 3):
        cx, cy = polar(400.0, 11.25 * k)
        bumps.append(
            Conductor(
                f"bump_{k:02d}",
                "Ground",
                to_lattice(rect(cx - 10, cy - 10, cx + 10, cy + 10)),
                z=BUMP_SPAN,
            )
        )
    s = Scene("cell11_bumps_flip_chip", (2, 11, 12), die, [])
    footprint = die_footprint(s)
    s.prisms = [
        PrismRecord(
            "substrate_d0",
            footprint,
            exact_z(-50),
            exact_z(0),
            "substrate",
            "dielectric",
        ),
        PrismRecord(
            "vacuum", footprint, exact_z(0), exact_z("8.2"), "vacuum", "vacuum"
        ),
        PrismRecord(
            "substrate_d1",
            footprint,
            exact_z("8.2"),
            exact_z("58.2"),
            "substrate",
            "dielectric",
        ),
    ]
    s.conductors = [
        Conductor("d0_ground", "Ground", to_lattice(rect(*die)), hole, D0_METAL),
        Conductor("d1_ground", "Ground", to_lattice(rect(*die)), hole, D1_METAL),
        *bumps,
    ]
    s.expectations = _flip_chip_expectations
    return s


def _flip_chip_expectations(scene: Scene) -> dict[str, Expectation]:
    """Each ground is banded on its face toward the gap (MA, less the bump contacts) and toward its own substrate
    (MS); each substrate surface (SA) from the ground on it.  The bumps have no gap edge and no band."""
    shapes = {c.entity: conductor_shape(c) for c in scene.conductors}
    bumps = unary_union([s for e, s in shapes.items() if e.startswith("bump")])
    um = 0.001
    clearance = box(*(lattice(v) * um for v in scene.die)).difference(
        shapes["d0_ground"]
    )
    out: dict[str, Expectation] = {}
    for die, gap_face, substrate_face, z in (
        ("d0", "top", "bottom", "0.0"),
        ("d1", "bottom", "top", "8.2"),
    ):
        metal = shapes[f"{die}_ground"]
        out[f"ma::{die}_ground::{gap_face}"] = (metal.difference(bumps), clearance)
        out[f"ms::{die}_ground::{substrate_face}"] = (metal, clearance)
        out[f"sa::substrate_{die}|vacuum::z={z}"] = (clearance, metal)
    return out


SCENES = {
    "cell01_disk": cell01_disk,
    "cell01_disk_test_build": lambda: cell01_disk(DISK_R_TEST_BUILD),
    "cell02_clearance": cell02_clearance,
    "disk_in_clearance": disk_in_clearance,
    "cell06_cpw_straight": cell06_cpw_straight,
    "cell07_cpw45_ro": lambda: cell07_cpw_45(RO_W, RO_G),
    "cell07_cpw45_xy": lambda: cell07_cpw_45(XY_W, XY_G),
    "ro_through_clearance": ro_through_clearance,
    "xy_tip": xy_tip,
    "d0_die": d0_die,
    "claw_floating": lambda: claw_scene(DISK_R, False),
    "claw_floating_test_build": lambda: claw_scene(DISK_R_TEST_BUILD, False),
    "claw_grounded": lambda: claw_scene(DISK_R, True),
    "junction_straight": lambda: junction_scene(False),
    "junction_fan": lambda: junction_scene(True),
    "junction_fan_noport": lambda: junction_scene(True, port=False),
    "d1_die_test_build": lambda: d1_die(False),
    "d1_die_defaults": lambda: d1_die(True),
    "cell11_bumps_flip_chip": cell11_bumps,
}
