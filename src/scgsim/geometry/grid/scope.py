"""The scope a polygon operation runs in, passed to it as an argument: the grid it nodes on, the frame its points are in, and what its refusals name.

``GridScope`` holds every fact besides the polygons themselves that changes what a grid
operation returns or what its refusal says. A caller builds one from the model grid it
works on (``GridScope(grid)``), places it in an occurrence's frame (:meth:`GridScope.placed`),
names the declared object it reads (:meth:`GridScope.naming`) or evaluates the unrounded
reference (:meth:`GridScope.as_reference`); every operation it calls receives it whole.
``UNGRIDDED`` is the scope of work on no model grid: AEDT's native planning and a junction's
off-axis local frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from scgsim.geometry.grid.grid_regions import GridPoint, grid_change_message
from scgsim.geometry.models.grid import GeometryGridContext, format_box
from scgsim.refusal import internal

# An affine map ``(a, b, c, d, x, y)``: a point ``(u, v)`` lands at ``(a u + b v + x, c u + d v + y)``.
Affine = tuple[float, ...]


@dataclass(frozen=True)
class NamedSource:
    """The declared object a recipe reads and the source layer its polygons are drawn on."""

    object_id: str
    layer: str


@dataclass(frozen=True)
class GridScope:
    """What a polygon operation reads besides its polygons.

    ``grid`` is the model grid the operation nodes on; without one a Boolean is gdstk's
    at its own precision. ``frame`` maps the polygons' own coordinates to the assembled
    layout, where the grid lies (None: they are already layout coordinates). ``reference``
    is true while a declared source recipe is evaluated as authored for its preservation
    audit: its Booleans overlay the operands' own coordinates and nothing is rounded.
    ``source`` is the declared object and layer a refusal of a drawn polygon names.
    """

    grid: GeometryGridContext | None
    frame: Affine | None = None
    reference: bool = False
    source: NamedSource | None = None

    def placed(self, transform: Sequence[float]) -> GridScope:
        """The same scope with the polygons in an occurrence's frame."""
        return replace(self, frame=tuple(transform))

    def naming(self, object_id: str, layer: str) -> GridScope:
        """The same scope with refusals naming ``object_id`` drawn on ``layer``."""
        return replace(self, source=NamedSource(object_id, layer))

    def as_reference(self) -> GridScope:
        """The same scope evaluating the recipe as authored, with nothing rounded."""
        return replace(self, reference=True)

    def bound_grid(self) -> GeometryGridContext:
        """The model grid of a source recipe, which runs only on one."""
        if self.grid is None:
            raise RuntimeError(
                internal(
                    "a source recipe ran without the layout grid; its GridScope must be built from the layout's grid"
                )
            )
        return self.grid

    @property
    def grid_size_um(self) -> float | None:
        """The grid step in micrometres, or None without a grid."""
        return None if self.grid is None else self.grid.grid_size_um

    def to_layout(self, point: Sequence[float]) -> GridPoint:
        """A point of the polygon frame in the assembled layout."""
        if self.frame is None:
            return tuple(point)
        a, b, c, d, x, y = self.frame
        u, v = point
        return a * u + b * v + x, c * u + d * v + y

    def from_layout(self, point: Sequence[float]) -> GridPoint:
        """A point of the assembled layout in the polygon frame."""
        if self.frame is None:
            return tuple(point)
        a, b, c, d, x, y = self.frame
        u, v = point
        determinant = a * d - b * c
        return (
            (d * (u - x) - b * (v - y)) / determinant,
            (-c * (u - x) + a * (v - y)) / determinant,
        )

    def place(self, points: Sequence[Sequence[float]]) -> str:
        """Where points of the polygon frame lie in the assembled layout, in micrometres."""
        world = [self.to_layout(point) for point in points]
        return format_box(
            (
                min(x for x, _ in world),
                min(y for _, y in world),
                max(x for x, _ in world),
                max(y for _, y in world),
            ),
            self.grid_size_um,
        )

    def refusal(self, lead: str, findings: list[str], action: str) -> str:
        """A drawn-polygon refusal naming the declared source, where it lies and who acts."""
        named = self.source
        return grid_change_message(
            lead if named is None else f"{lead}: {named.object_id}",
            [] if named is None else [named.layer],
            self.grid_size_um,
            findings,
            action,
        )


UNGRIDDED = GridScope(None)
