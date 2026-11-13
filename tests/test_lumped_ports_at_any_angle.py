"""A lumped port at any angle is the lattice rectangle of its drawn sheet (issue 165).

On the 1 nm grid, a rectangle whose corners are grid points runs along an integer direction
v = (p, q); the partition cuts it every |v| grid steps, so each cut ends on grid points.  Each
layout here is drawn in the frame of such a v: a point is s v + t w with w = (-q, p) and whole
s and t, so every vertex is a grid point.  Each goes through ``EigenmodeSim``'s own preparation
or through the AEDT preparation, and where a claim is about the built model through the SGB
geometry build that ``simulation_drc`` and ``mesh()`` share.  No mesher or solver runs.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

import pytest
from scgsim.geometry.planning.junction_partition import lattice_rectangle
from test_palace_rectangular_ports import VACUUM_PADDING_UM, port_group, prepared_port
from test_simulation_drc import MATERIALS, NO_VACUUM_STACK

from scgsim.geometry import GeometryPlan
from scgsim.palace import EigenmodeSim

GRID_UM = 0.001
NM_PER_UM = 1000
PORT = "chip/device/P1"
NETS = {"pad": ("chip/device/pad",), "electrode": ("chip/device/electrode",)}
# Half a grid diagonal: the farthest the grid moves a point it rounds.
SNAP_BOUND_UM = math.sqrt(2) / 2 * GRID_UM

type Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class LatticeLayout:
    """A pad, an electrode beside it and a lumped-port sheet between them, along v = (p, q).

    Each box is ``(s0, s1, t0, t1)`` in steps of v along the port and of w = (-q, p) across
    it.  The pad ends at s = 0 and the electrode starts at the electrode's s0, both with an edge
    at right angles to the sheet.  With ``bent_sides`` the sheet has one more vertex on each
    long side, in the middle of the gap: on the side t0 it lies on the side, on the side t1 it
    lies half a grid diagonal outside, (0, 1) grid steps off a side along (1, -1) or (1, 1).
    """

    direction: tuple[int, int]
    pad: Box
    electrode: Box
    sheet: Box
    bent_sides: bool = False

    def steps(self, s: int, t: int) -> tuple[int, int]:
        """The grid point s v + t w, in grid steps."""
        p, q = self.direction
        return (s * p - t * q, s * q + t * p)

    def point_um(self, s: int, t: int) -> tuple[float, float]:
        """The grid point s v + t w, in micrometres."""
        x, y = self.steps(s, t)
        return (x / NM_PER_UM, y / NM_PER_UM)

    def box_um(self, box: Box) -> list[tuple[float, float]]:
        """The box's four corners in micrometres, in ring order."""
        s0, s1, t0, t1 = box
        return [
            self.point_um(s, t) for s, t in ((s0, t0), (s1, t0), (s1, t1), (s0, t1))
        ]

    def sheet_um(self) -> list[tuple[float, float]]:
        """The sheet's ring, with its two added vertices when ``bent_sides``."""
        s0, s1, t0, t1 = self.sheet
        if not self.bent_sides:
            return self.box_um(self.sheet)
        middle = self.electrode[0] // 2
        x, y = self.steps(middle, t1)
        return [
            self.point_um(s0, t0),
            self.point_um(middle, t0),
            self.point_um(s1, t0),
            self.point_um(s1, t1),
            (x / NM_PER_UM, (y + 1) / NM_PER_UM),
            self.point_um(s0, t1),
        ]

    def centre_um(self, box: Box) -> tuple[float, float]:
        """The box's centre in micrometres."""
        s0, s1, t0, t1 = box
        x, y = self.steps(s0 + s1, t0 + t1)
        return (x / (2 * NM_PER_UM), y / (2 * NM_PER_UM))

    @property
    def spacing_um(self) -> float:
        """The cut spacing |v| g."""
        return math.hypot(*self.direction) * GRID_UM


# A 45 degree sheet 0.283 um wide across a 1.980 um gap, drawn along (1, -1) so that an
# instance turned by a quarter turn places it along (1, 1).
BENT_45 = LatticeLayout(
    direction=(1, -1),
    pad=(-7000, 0, -7000, 7000),
    electrode=(1400, 2400, -1000, 1000),
    sheet=(-200, 1600, -100, 100),
    bent_sides=True,
)
# A sheet along (3, 4): cut every 5 grid steps, 0.4 um wide across a 2 um gap.
ALONG_3_4 = LatticeLayout(
    direction=(3, 4),
    pad=(-2000, 0, -2000, 2000),
    electrode=(400, 600, -400, 400),
    sheet=(-40, 440, -40, 40),
)
# A 100 um square pad, an electrode from x = 60 um and an 11 x 1 um sheet from x = 49.5 um, all
# drawn along x once moved by ALONG_X_OFFSET_UM; through an instance turned by any other angle
# than a quarter turn, the sheet's corners leave the grid.
ALONG_X = LatticeLayout(
    direction=(1, 0),
    pad=(-100000, 0, -50000, 50000),
    electrode=(10000, 25000, -15000, 15000),
    sheet=(-500, 10500, -500, 500),
)
ALONG_X_OFFSET_UM = 50.0
# A 45 degree sheet 200 grid steps long and 10 wide, its corners A (0, 0), B (200, 200),
# C (190, 210) and D (-10, 10) in grid steps, crossing a gap from s = 50 to s = 150.
NEAR_CORNER_45 = LatticeLayout(
    direction=(1, 1),
    pad=(-1000, 50, -500, 500),
    electrode=(150, 1000, -500, 500),
    sheet=(0, 200, 0, 10),
)
# The Xmon's port at a small scale: a 45 degree sheet 190 grid steps wide, from a round pad (the
# circle inscribed in the box ``pad``, 0.99985 um in radius, centred on the sheet's middle line) to
# an electrode 2800 grid steps along it.
ROUND_PAD_45 = LatticeLayout(
    direction=(1, 1),
    pad=(-714, 700, -612, 802),
    electrode=(2800, 5000, -1500, 1700),
    sheet=(0, 4000, 0, 190),
)
# Substrate-air bands 0.5 and 1 um wide, as the Xmon tutorial asks for.
SA_BANDS = {
    "SA": {"film_thickness_m": 2e-9, "permittivity": 4, "inset_margins_um": [0.5, 1.0]}
}


def lattice_device(monkeypatch, layout: LatticeLayout, offset_um: float = 0.0):
    """The layout as a gdsfactory component, moved along x by ``offset_um``, with its port P1."""

    def moved(points):
        return [(x + offset_um, y) for x, y in points]

    _, _, t0, t1 = layout.sheet
    return device_with_port(
        monkeypatch,
        pad=moved(layout.box_um(layout.pad)),
        electrode=moved(layout.box_um(layout.electrode)),
        sheet=moved(layout.sheet_um()),
        port_centre=moved([layout.centre_um(layout.sheet)])[0],
        port_width_um=(t1 - t0) * layout.spacing_um,
        direction=layout.direction,
    )


def device_with_port(  # noqa: PLR0913 -- the three polygons and the port that make the device
    monkeypatch, *, pad, electrode, sheet, port_centre, port_width_um, direction
):
    """A pad, an electrode and a lumped-port sheet P1 pointing along ``direction``, as a gdsfactory component."""
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    monkeypatch.setattr(
        gf.pdk,
        "_ACTIVE_PDK",
        gf.pdk._ACTIVE_PDK,  # noqa: SLF001 -- preserve the active PDK during test setup
    )
    gf.gpdk.PDK.activate()
    device = gf.Component()
    device.add_polygon(pad, layer=(1, 0))
    device.add_polygon(electrode, layer=(2, 0))
    device.add_polygon(sheet, layer=(9, 0))
    # gdsfactory keeps a port width in whole pairs of grid steps; SCGSim reads the sheet itself.
    width_nm = port_width_um * NM_PER_UM
    device.add_port(
        name="P1",
        center=port_centre,
        width=2 * round(width_nm / 2) / NM_PER_UM,
        orientation=math.degrees(math.atan2(direction[1], direction[0])),
        layer=(9, 0),
    )

    def selector(points):
        xs, ys = [x for x, _ in points], [y for _, y in points]
        return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)

    device.info["component_semantics"] = {
        "schema_version": 2,
        "conductor_regions": [
            {
                "semantic_id": name,
                "level": "metal",
                "gds_layer": [layer, 0],
                "geometry": {
                    "geometry_source": "gds_polygon",
                    "selector_point_um": selector(points),
                },
            }
            for name, layer, points in (("pad", 1, pad), ("electrode", 2, electrode))
        ],
        "ports": [{"name": "P1", "layer": [9, 0], "target_layer": "metal"}],
    }
    return device


def placed(device, rotation_degrees: float):
    """A chip holding the device through an instance turned by ``rotation_degrees``, inside an undeclared die outline."""
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    chip = gf.Component()
    reference = chip << device
    reference.rotate(rotation_degrees)
    box = chip.dbbox()
    chip.add_polygon(
        [
            (box.left - 10, box.bottom - 10),
            (box.right + 10, box.bottom - 10),
            (box.right + 10, box.top + 10),
            (box.left - 10, box.top + 10),
        ],
        layer=(50, 0),
    )
    chip.info["component_semantics"] = {"schema_version": 2, "conductor_regions": []}
    return chip, reference


def plan_snapshot(chip, reference, *, outer_vacuum_um: float | None = None):
    """The prepared plan of the chip, its device registered as ``chip/device``.

    ``outer_vacuum_um`` adds the plan's own vacuum region that far above the layout, as the
    AEDT preparation needs.
    """
    plan = GeometryPlan(
        chip,
        root_id="chip",
        layer_stack=NO_VACUUM_STACK,
        material_records=MATERIALS,
        coupon_padding_um=0.0,
    )
    plan.register_instance("chip/device", reference)
    plan.set_nets(NETS)
    if outer_vacuum_um is not None:
        plan.set_vacuum_region(padding=(0.0, 0.0, outer_vacuum_um))
    return plan.prepare()


def eigenmode_sim(chip, reference):
    """An ``EigenmodeSim`` on the chip's plan with its one lumped port, as a user sets it up."""
    sim = EigenmodeSim()
    sim.set_plan(plan_snapshot(chip, reference))
    sim.set_vacuum_region(VACUUM_PADDING_UM)
    sim.add_port(PORT, layer="metal", layout_sheet=True, inductance=1e-9)
    return sim


def round_pad(layout: LatticeLayout, count: int) -> list[tuple[float, float]]:
    """The circle inscribed in the layout's pad box, drawn as ``count`` vertices on the grid, in micrometres."""
    s0, s1, t0, t1 = layout.pad
    cx, cy = layout.centre_um(layout.pad)
    radius_um = (s1 - s0) / 2 * layout.spacing_um
    assert s1 - s0 == t1 - t0
    return [
        (
            round((cx + radius_um * math.cos(2 * math.pi * k / count)) * NM_PER_UM)
            / NM_PER_UM,
            round((cy + radius_um * math.sin(2 * math.pi * k / count)) * NM_PER_UM)
            / NM_PER_UM,
        )
        for k in range(count)
    ]


def built_port_area(monkeypatch, chip, reference) -> float:
    """The port face area of the SGB build ``simulation_drc`` makes."""
    return port_group(built_sidecar(monkeypatch, chip, reference))["area_um2"]


def built_sidecar(monkeypatch, chip, reference, **options):
    """The sidecar of the SGB build ``simulation_drc`` makes, read before its folder is removed."""
    sidecars = []
    real = EigenmodeSim._build_geometry  # noqa: SLF001 -- the geometry half the check runs

    def recording(self, output_dir):
        geometry = real(self, output_dir)
        sidecars.append(json.loads(geometry.artifacts.sidecar.read_bytes()))
        return geometry

    monkeypatch.setattr(EigenmodeSim, "_build_geometry", recording)
    assert EigenmodeSim.simulation_drc(
        chip,
        layer_stack=NO_VACUUM_STACK,
        material_records=MATERIALS,
        instances={"chip/device": reference},
        nets=NETS,
        **options,
    ).passed
    (sidecar,) = sidecars
    return sidecar


def turned(point, degrees: float) -> tuple[float, float]:
    """A point turned about the origin, as the instance turns it."""
    angle = math.radians(degrees)
    x, y = point
    return (
        x * math.cos(angle) - y * math.sin(angle),
        x * math.sin(angle) + y * math.cos(angle),
    )


def cut_ends(lattice) -> set[tuple[float, float]]:
    """The four ends of the two recorded cuts: the corners of the central rectangle."""
    return {tuple(end) for cut in lattice["cuts"] for end in cut["ends_um"]}


def test_a_45_degree_port_with_bent_sides_is_its_central_lattice_rectangle(monkeypatch):
    """A 45 degree sheet bent by its two added vertices: the port is still the exact rectangle.

    The vertex on one long side lies on it; the vertex on the other lies half a grid diagonal
    off, the snap bound, so both are dropped and the sheet is read as its four corners.  The
    cuts lie on the two straight conductor edges, every sqrt 2 grid steps along v = (1, 1).
    """
    layout = BENT_45
    chip, reference = placed(lattice_device(monkeypatch, layout), 90)
    _, region, record = prepared_port(eigenmode_sim(chip, reference))
    lattice = record["lattice"]
    assert lattice["direction_grid"] == [1, 1]
    assert lattice["cut_spacing_um"] == pytest.approx(math.sqrt(2) * GRID_UM)
    assert region.metadata["rectangular_junction_lattice"] == lattice
    # A quarter turn keeps the grid: the corners are the drawn ones.
    assert all(
        abs(value) < 1e-12 for shift in lattice["corner_shifts_um"] for value in shift
    )
    s0, _, t0, t1 = layout.sheet
    gap_end = layout.electrode[0]
    assert [cut["index"] for cut in lattice["cuts"]] == [-s0, gap_end - s0]
    assert [cut["from_corner_a_um"] for cut in lattice["cuts"]] == pytest.approx(
        [-s0 * layout.spacing_um, (gap_end - s0) * layout.spacing_um]
    )
    expected = {
        tuple(round(v, 9) for v in turned(layout.point_um(s, t), 90))
        for s in (0, gap_end)
        for t in (t0, t1)
    }
    assert {tuple(round(v, 9) for v in p) for p in region.exterior} == expected
    assert cut_ends(lattice) == {tuple(map(float, p)) for p in region.exterior}
    length_um, width_um = gap_end * layout.spacing_um, (t1 - t0) * layout.spacing_um
    assert built_port_area(monkeypatch, chip, reference) == pytest.approx(
        length_um * width_um, abs=1e-9
    )


def test_a_vertex_beside_a_corner_is_dropped_and_the_corner_kept(monkeypatch):
    """A vertex (1, 0) drawn just after corner A, half a grid diagonal off side AB, is dropped.

    A in turn lies within half a grid diagonal of the line from D to that vertex, so the sheet's
    vertices hold two sets of four corners by the rule; only A, B, C and D are a rectangle with
    its corners on the grid, so they are the corners, and the port is cut from them.
    """
    layout = NEAR_CORNER_45
    a, b, c, d = layout.box_um(layout.sheet)
    sheet = [a, (1 / NM_PER_UM, 0.0), b, c, d]
    device = device_with_port(
        monkeypatch,
        pad=layout.box_um(layout.pad),
        electrode=layout.box_um(layout.electrode),
        sheet=sheet,
        port_centre=layout.centre_um(layout.sheet),
        port_width_um=10 * layout.spacing_um,
        direction=layout.direction,
    )
    chip, reference = placed(device, 0)
    _, region, record = prepared_port(eigenmode_sim(chip, reference))
    lattice = record["lattice"]
    assert lattice["corners_grid"] == [[0, 0], [200, 200], [190, 210], [-10, 10]]
    assert lattice["direction_grid"] == [1, 1]
    assert [cut["index"] for cut in lattice["cuts"]] == [50, 150]
    expected = {layout.point_um(s, t) for s in (50, 150) for t in (0, 10)}
    assert {tuple(map(float, p)) for p in region.exterior} == expected


def test_two_sets_of_corners_equally_near_a_rectangle_are_refused():
    """A sheet drawn with each corner doubled one grid step along x holds two rectangles exactly.

    Read with the corners (0, 0), (200, 200), (190, 210) and (-10, 10), in grid steps, every
    other vertex lies within half a grid diagonal of a side, and read with their twins one step
    along x as well.  Neither moves a corner, so the drawing does not say where the port lies,
    and the sheet is refused with both.  It is read as drawn: placed through a plan, the grid's
    snap rounding drops the corner (-10, 10) and leaves only the twins.
    """
    corners = [(0, 0), (200, 200), (190, 210), (-10, 10)]
    twins = [(x + 1, y) for x, y in corners]
    (a, b, c, d), (a1, b1, c1, d1) = corners, twins
    ring = [(x / NM_PER_UM, y / NM_PER_UM) for x, y in (a, a1, b, b1, c1, c, d1, d)]
    with pytest.raises(
        ValueError, match="more than one set of four corners"
    ) as refused:
        lattice_rectangle(
            ring,
            GRID_UM,
            (math.sqrt(0.5), math.sqrt(0.5)),
            drawn=ring,
            port=f"junction {PORT!r}",
            layer="9/0",
        )
    message = str(refused.value)
    assert message.startswith(
        f"junction {PORT!r} has more than one set of four corners equally near a "
        "rectangle with its corners on the grid on layer 9/0 (grid g = 0.001 um): "
    )
    rectangles = [
        {
            (round(float(x) * NM_PER_UM), round(float(y) * NM_PER_UM))
            for x, y in re.findall(r"\(([-\d.e]+), ([-\d.e]+)\) um", found)
        }
        for found in re.findall(
            r"it is the rectangle with its corners at ([^;]+)", message
        )
    ]
    assert sorted(map(sorted, rectangles)) == [sorted(corners), sorted(twins)]
    assert message.endswith(
        "Fix the layout: draw each corner of the sheet as one vertex."
    )


def test_a_port_along_3_4_is_cut_every_5_grid_steps(monkeypatch):
    """Along v = (3, 4), |v| = 5: the cuts are 5 nm apart and end on grid points, and the port is the rectangle between the conductors."""
    layout = ALONG_3_4
    chip, reference = placed(lattice_device(monkeypatch, layout), 0)
    _, region, record = prepared_port(eigenmode_sim(chip, reference))
    lattice = record["lattice"]
    assert lattice["direction_grid"] == [3, 4]
    assert lattice["cut_spacing_um"] == pytest.approx(5 * GRID_UM)
    s0, _, t0, t1 = layout.sheet
    gap_end = layout.electrode[0]
    assert [cut["index"] for cut in lattice["cuts"]] == [-s0, gap_end - s0]
    expected = {layout.point_um(s, t) for s in (0, gap_end) for t in (t0, t1)}
    assert {tuple(map(float, p)) for p in region.exterior} == expected
    assert cut_ends(lattice) == expected
    assert built_port_area(monkeypatch, chip, reference) == pytest.approx(
        gap_end * layout.spacing_um * (t1 - t0) * layout.spacing_um, abs=1e-9
    )


def test_a_sheet_whose_rounded_corners_are_no_rectangle_is_shifted_onto_the_nearest(
    monkeypatch,
):
    """A sheet drawn along x, placed through a 45 degree instance: the grid rounds its corners off a rectangle.

    SCGSim takes the nearest rectangle with grid-point corners each within half a grid
    diagonal of the drawn (turned) corner, and records each corner's shift from it.  The grid
    rounds the corner B to (43.134, 42.426) um; the rectangle puts it at (43.133, 42.426) um,
    0.65 nm from where it was drawn.
    """
    layout = ALONG_X
    device = lattice_device(monkeypatch, layout, offset_um=ALONG_X_OFFSET_UM)
    chip, reference = placed(device, 45)
    _, region, record = prepared_port(eigenmode_sim(chip, reference))
    lattice = record["lattice"]
    assert lattice["direction_grid"] == [1, 1]
    drawn = [
        turned((x + ALONG_X_OFFSET_UM, y), 45) for x, y in layout.box_um(layout.sheet)
    ]
    assert lattice["drawn_corners_um"] == [
        pytest.approx(list(p), abs=1e-12) for p in drawn
    ]
    assert lattice["corners_grid"] == [
        [35355, 34648],
        [43133, 42426],
        [42426, 43133],
        [34648, 35355],
    ]
    for corner, original, shift in zip(
        lattice["corners_um"],
        lattice["drawn_corners_um"],
        lattice["corner_shifts_um"],
        strict=True,
    ):
        assert shift == pytest.approx(
            [corner[0] - original[0], corner[1] - original[1]], abs=1e-14
        )
        assert math.hypot(*shift) <= SNAP_BOUND_UM
    rounded = [[round(v / GRID_UM) for v in point] for point in drawn]
    assert rounded[1] == [43134, 42426]
    assert lattice["corners_grid"][1] != rounded[1]
    assert [cut["index"] for cut in lattice["cuts"]] == [354, 7425]
    assert cut_ends(lattice) == {tuple(map(float, p)) for p in region.exterior}
    width_um = lattice["width_steps"] * lattice["cut_spacing_um"]
    length_um = (7425 - 354) * lattice["cut_spacing_um"]
    assert built_port_area(monkeypatch, chip, reference) == pytest.approx(
        length_um * width_um, abs=1e-9
    )


@pytest.mark.parametrize(
    ("rotation", "finding"),
    [
        (10, "no rectangle with its corners on grid points has each corner within"),
        (
            30,
            r"runs along v = \(433, 250\), not along the port direction \(0.866025, 0.5\)",
        ),
    ],
    ids=["none-near", "not-along-the-port"],
)
def test_a_sheet_no_lattice_rectangle_fits_is_refused_with_its_corners(
    monkeypatch, rotation, finding
):
    """Through a 10 or 30 degree instance no rectangle on the grid fits near the drawn corners along the port.

    The refusal names the port, its layer and grid, its four drawn corners in um and the fix.
    """
    layout = ALONG_X
    device = lattice_device(monkeypatch, layout, offset_um=ALONG_X_OFFSET_UM)
    chip, reference = placed(device, rotation)
    sim = eigenmode_sim(chip, reference)
    with pytest.raises(ValueError, match=finding) as refused:
        prepared_port(sim)
    message = str(refused.value)
    assert message.startswith(
        f"lumped port {PORT!r} has no rectangle with its corners on the grid near its "
        "drawn corners on layer 9/0 (grid g = 0.001 um): its corners are drawn at "
    )
    corners = [
        (float(x), float(y))
        for x, y in re.findall(r"\(([-\d.e]+), ([-\d.e]+)\) um", message)
    ]
    drawn = [
        turned((x + ALONG_X_OFFSET_UM, y), rotation)
        for x, y in layout.box_um(layout.sheet)
    ]
    assert sorted(corners) == [pytest.approx(p, abs=1e-12) for p in sorted(drawn)]
    assert "Fix the layout: draw the sheet with its four corners on the grid" in message


def test_a_gap_shorter_than_the_cut_spacing_is_refused_where_it_lies(monkeypatch):
    """Along v = (1000, 1) a cut lies only every 1 um: a 0.3 um gap holds none, and the refusal locates the gap.

    The sheet runs from x = -1 um to 2 um; the pad ends at x = 0.2 um and the electrode starts
    at x = 0.5 um, between the cut positions at 0 and 1 um along the sheet.
    """
    device = device_with_port(
        monkeypatch,
        pad=[(-20, -10), (0.2, -10), (0.2, 10), (-20, 10)],
        electrode=[(0.5, -10), (20, -10), (20, 10), (0.5, 10)],
        sheet=[(-1.0, 0.0), (2.0, 0.003), (1.999, 1.003), (-1.001, 1.0)],
        port_centre=(0.5, 0.5),
        port_width_um=1.0,
        direction=(1000, 1),
    )
    chip, reference = placed(device, 0)
    with pytest.raises(ValueError, match="has no room for a rectangle") as refused:
        prepared_port(eigenmode_sim(chip, reference))
    message = str(refused.value)
    assert "PORT_SHEET__chip/device/P1" in message
    assert "on layer 9/0 (grid g = 0.001 um)" in message
    assert "runs along v = (1000, 1), where a cut lies only every 1000 nm" in message
    assert "runs from (0.200, 0.501) um to (0.500, 0.502) um" in message
    assert "Fix the layout: lengthen the gap the sheet crosses" in message


def test_a_45_degree_port_on_a_round_pad_keeps_every_band_boundary(monkeypatch):
    """Substrate-air bands 0.5 and 1 um wide beside a 45 degree port landing on a round pad of 128 vertices.

    The partition joins an end piece to the pad.  Its corner at the cut lies tens of nm from the
    next pad vertices, and the 1 um arcs round them meet at a shallow angle, where snap rounding
    makes a few edges common to both.  Each such edge is one arc's seam, so the two seams meet at
    one point and the level arrangement keeps every band boundary.
    """
    layout = ROUND_PAD_45
    _, _, t0, t1 = layout.sheet
    device = device_with_port(
        monkeypatch,
        pad=round_pad(layout, 128),
        electrode=layout.box_um(layout.electrode),
        sheet=layout.box_um(layout.sheet),
        port_centre=layout.centre_um(layout.sheet),
        port_width_um=(t1 - t0) * layout.spacing_um,
        direction=layout.direction,
    )
    chip, reference = placed(device, 0)
    sidecar = built_sidecar(monkeypatch, chip, reference, surface_epr_specs=SA_BANDS)
    bands = {
        name.rsplit("::band=", 1)[1]
        for name in sidecar["group"]
        if name.startswith("SA::") and "::band=" in name
    }
    assert bands == {"0", "1", "2"}


def test_aedt_binds_a_45_degree_sheet_to_its_central_lattice_rectangle(monkeypatch):
    """The AEDT preparation reads the bent 45 degree sheet as Palace does: the same rectangle, v = (1, 1)."""
    from scgsim.aedt import (  # noqa: PLC0415 -- the optional AEDT adapter, loaded only for its own check
        planar_junction_from_port,
        prepare_planar_geometry_input,
    )

    layout = BENT_45
    chip, reference = placed(lattice_device(monkeypatch, layout), 90)
    snapshot = plan_snapshot(chip, reference, outer_vacuum_um=10.0)
    junction = planar_junction_from_port(
        snapshot.geometry_input,
        port_name=PORT,
        junction_id="jj",
        terminal_a_net="pad",
        terminal_b_net="electrode",
        inductance_h=1e-8,
        capacitance_f=0.0,
    )
    s0, _, t0, t1 = layout.sheet
    assert junction.direction_xy == pytest.approx((math.sqrt(0.5), math.sqrt(0.5)))
    assert junction.width_um == pytest.approx((t1 - t0) * layout.spacing_um)
    prepared = prepare_planar_geometry_input(
        snapshot.geometry_input,
        prepared_stack=snapshot.stack,
        modeling="solid",
        junctions=(junction,),
    )
    record = prepared.source["junction_partitions"]["jj"]
    # The AEDT source keeps its records as tuples.
    assert list(record["lattice"]["direction_grid"]) == [1, 1]
    ((central,),) = record["central_model_regions"]
    gap_end = layout.electrode[0]
    expected = {
        tuple(round(v, 9) for v in turned(layout.point_um(s, t), 90))
        for s in (0, gap_end)
        for t in (t0, t1)
    }
    assert {tuple(round(float(v), 9) for v in p) for p in central} == expected
