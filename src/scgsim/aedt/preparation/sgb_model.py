"""Prepare the SGB model AEDT imports: the guarded STEP of one SGB build and what AEDT assigns on it.

AEDT models the geometry Palace meshes (issue #67).  One preparation runs, in order:

1. the physical interpretation of the source under the requested ``modeling``, unchanged
   (:mod:`scgsim.aedt.preparation.modeling`), which also derives the plan's rectangular junctions into
   their port sheets;
2. the binding of each AEDT junction to the port sheet of the plan's ``RectangularJunctionSpec`` of the
   same id;
3. the SGB build with the Surface-EPR bands of the normalized specs
   (:func:`scgsim.geometry.surface_epr.build_banded_geometry`).  ``thin_film`` builds Route A (conductor
   sheets); ``solid`` builds Route C (conductor solids), which is how AEDT realizes a Route B request
   (decision 7): the same faces and bands, with the conductor interiors kept as solids;
4. :func:`scgsim.geometry.sgb.exchange.step.export_aedt_step`, which writes the STEP without the vacuum and
   its durable guard.  AEDT creates its own Region from the plan's ``VACUUM_REGION`` bounds instead.

The result, :class:`PreparedSgbModel`, binds the files by SHA-256 so the runtime checks them again before
it imports anything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scgsim.aedt.data import canonical_sha256, detached, freeze
from scgsim.aedt.preparation.modeling import interpret_geometry
from scgsim.aedt.specs.epr import PlanarJunction
from scgsim.aedt.specs.modeling import Modeling, modeling_from_value
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.sgb.exchange.guard import file_sha256, read_step_guard
from scgsim.geometry.sgb.exchange.step import export_aedt_step
from scgsim.geometry.sgb.sidecar.files import read_sidecar
from scgsim.geometry.surface_epr import build_banded_geometry

SCHEMA = "scgsim.aedt.sgb-model.v1"
VACUUM_REGION = "VACUUM_REGION"
STEP_NAME = "model.step"
REQUESTED_ROUTE: dict[str, Literal["A", "B"]] = {"thin_film": "A", "solid": "B"}
REALIZED_ROUTE: dict[str, Literal["A", "C"]] = {"thin_film": "A", "solid": "C"}
_BOUND_FILES = ("step", "geometry", "sidecar", "guard")


@dataclass(frozen=True)
class SgbJunction:
    """One AEDT junction and the lumped-port group (``PORT::<port sheet>``) of the SGB model it sits on."""

    junction: PlanarJunction
    port_group: str

    def to_payload(self) -> dict[str, Any]:
        """The canonical payload of this binding."""
        return {"junction": self.junction.to_payload(), "port_group": self.port_group}

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> SgbJunction:
        """Decode a binding written by :meth:`to_payload`."""
        return cls(PlanarJunction.from_payload(value["junction"]), value["port_group"])


@dataclass(frozen=True)
class PreparedSgbModel:
    """The bound STEP, sidecar and guard of one SGB build, and the AEDT inputs bound to its groups.

    ``model_dir`` holds the files; ``files`` names each relative to it with its SHA-256 (``step``,
    ``geometry`` the BREP the STEP was written from, ``sidecar`` and ``guard``).  ``model_identity`` is
    the sidecar's model record.  ``vacuum_bounds_um`` is ``(x_min, y_min, z_min, x_max, y_max, z_max)`` of
    the plan's ``VACUUM_REGION``, the box SGB built and the box AEDT's Region takes.  ``materials`` are the
    stack's records of the materials the model's volumes name, which AEDT realizes on the imported objects.
    """

    modeling: Modeling
    requested_route: Literal["A", "B"]
    route: Literal["A", "C"]
    model_dir: Path
    files: Mapping[str, Mapping[str, str]]
    model_identity: Mapping[str, str]
    tolerance_um: float
    vacuum_bounds_um: tuple[float, float, float, float, float, float]
    materials: Mapping[str, Mapping[str, Any]]
    junctions: tuple[SgbJunction, ...] = ()
    surface_epr_specs: Mapping[str, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        modeling_from_value(self.modeling)
        if (
            self.requested_route != REQUESTED_ROUTE[self.modeling]
            or self.route != REALIZED_ROUTE[self.modeling]
        ):
            raise ValueError(
                f"modeling {self.modeling!r} is Route {REQUESTED_ROUTE[self.modeling]} realized as "
                f"Route {REALIZED_ROUTE[self.modeling]}"
            )
        if set(self.files) != set(_BOUND_FILES):
            raise ValueError(f"a prepared SGB model binds exactly {_BOUND_FILES}")
        object.__setattr__(self, "files", freeze(self.files))
        object.__setattr__(self, "model_identity", freeze(self.model_identity))
        object.__setattr__(self, "materials", freeze(self.materials))
        object.__setattr__(self, "junctions", tuple(self.junctions))
        if self.surface_epr_specs is not None:
            object.__setattr__(
                self, "surface_epr_specs", freeze(self.surface_epr_specs)
            )

    def path(self, role: str) -> Path:
        """The file bound under ``role`` (``step``, ``geometry``, ``sidecar`` or ``guard``)."""
        return self.model_dir / self.files[role]["path"]

    @property
    def identity_sha256(self) -> str:
        """The digest of everything this record binds; saved analyses compare it."""
        payload = self.to_payload()
        payload.pop("model_dir")
        return canonical_sha256(payload)

    def verify_files(self) -> None:
        """Raise ``ValueError`` when a bound file is missing or changed since preparation."""
        for role in _BOUND_FILES:
            path = self.path(role)
            if not path.is_file() or file_sha256(path) != self.files[role]["sha256"]:
                raise ValueError(
                    f"prepared SGB model {role} file {path.name} is missing or changed"
                )
        read_step_guard(self.path("guard"))

    def to_payload(self) -> dict[str, Any]:
        """The canonical payload of this record (round-trips through :meth:`from_payload`)."""
        return {
            "schema_version": SCHEMA,
            "modeling": self.modeling,
            "requested_route": self.requested_route,
            "route": self.route,
            "model_dir": self.model_dir.as_posix(),
            "files": detached(self.files),
            "model_identity": detached(self.model_identity),
            "tolerance_um": self.tolerance_um,
            "vacuum_bounds_um": list(self.vacuum_bounds_um),
            "materials": detached(self.materials),
            "junctions": [item.to_payload() for item in self.junctions],
            "surface_epr_specs": detached(self.surface_epr_specs),
        }

    @classmethod
    def from_payload(cls, value: Any) -> PreparedSgbModel:
        """Decode a record written by :meth:`to_payload`; a payload of another schema is refused."""
        if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} payload")
        bounds = tuple(float(v) for v in value["vacuum_bounds_um"])
        if len(bounds) != 6:
            raise ValueError("vacuum_bounds_um must hold six numbers")
        return cls(
            modeling=value["modeling"],
            requested_route=value["requested_route"],
            route=value["route"],
            model_dir=Path(value["model_dir"]),
            files=value["files"],
            model_identity=value["model_identity"],
            tolerance_um=float(value["tolerance_um"]),
            vacuum_bounds_um=(
                bounds[0],
                bounds[1],
                bounds[2],
                bounds[3],
                bounds[4],
                bounds[5],
            ),
            materials=value["materials"],
            junctions=tuple(
                SgbJunction.from_payload(item) for item in value["junctions"]
            ),
            surface_epr_specs=value["surface_epr_specs"],
        )


def prepare_sgb_model(  # noqa: PLR0913 -- the source, its stack and the AEDT model inputs are independent
    build_input: GeometryBuildInput,
    *,
    prepared_stack: Mapping[str, Any],
    modeling: Modeling,
    output_dir: str | Path,
    junctions: Sequence[PlanarJunction] = (),
    surface_epr_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> PreparedSgbModel:
    """Build the SGB model of ``build_input`` for AEDT into ``output_dir`` and bind its files.

    ``surface_epr_specs`` are Palace's ``MA``/``MS``/``SA`` specs after
    :func:`scgsim.geometry.surface_epr.normalize_surface_epr_specs`; their margins become bands.
    """
    modeling = modeling_from_value(modeling)
    if not isinstance(build_input, GeometryBuildInput):
        raise TypeError("build_input must be GeometryBuildInput")
    if build_input.lumped_supports:
        # TODO(human): issue #67 decision 8 settles how lumped supports reach the SGB model.
        raise NotImplementedError(
            "the SGB model has no lumped supports; see issue #67 decision 8"
        )
    effective_input, effective_stack, plan = interpret_geometry(
        build_input, prepared_stack, modeling
    )
    bounds = vacuum_bounds_um(effective_stack)
    bound_junctions = bind_junctions(effective_input, junctions)
    model_dir = Path(output_dir).expanduser().resolve()
    artifacts = build_banded_geometry(
        effective_input,
        plan.context,
        route=REALIZED_ROUTE[modeling],
        surface_epr_specs=surface_epr_specs,
        output_dir=model_dir,
    )
    exported = export_aedt_step(
        artifacts.brep, artifacts.sidecar, model_dir / STEP_NAME
    )
    sidecar = read_sidecar(artifacts.sidecar)
    check_region_box(sidecar, bounds)
    _check_junction_ports(sidecar, bound_junctions)
    files = {
        role: {"path": path.name, "sha256": file_sha256(path)}
        for role, path in (
            ("step", exported.path),
            ("geometry", artifacts.brep),
            ("sidecar", artifacts.sidecar),
            ("guard", exported.guard_path),
        )
    }
    return PreparedSgbModel(
        modeling=modeling,
        requested_route=REQUESTED_ROUTE[modeling],
        route=REALIZED_ROUTE[modeling],
        model_dir=model_dir,
        files=files,
        model_identity=sidecar["model"],
        tolerance_um=float(sidecar["tolerance_um"]),
        vacuum_bounds_um=bounds,
        materials=model_materials(sidecar, effective_stack),
        junctions=bound_junctions,
        surface_epr_specs=surface_epr_specs,
    )


def vacuum_bounds_um(
    stack: Mapping[str, Any],
) -> tuple[float, float, float, float, float, float]:
    """``(x_min, y_min, z_min, x_max, y_max, z_max)`` of the stack's ``VACUUM_REGION``.

    AEDT has no vacuum solid of its own: its Region is this box, so a stack without the plan's
    ``set_vacuum_region`` is refused.
    """
    region = stack.get("solution_regions", {}).get(VACUUM_REGION)
    if not isinstance(region, Mapping):
        raise ValueError(
            "AEDT creates its Region from the plan's VACUUM_REGION; call "
            "GeometryPlan.set_vacuum_region(...) before prepare()"
        )
    geometry = region["geometry"]
    xy = geometry["domain_bounds_um"]
    return (
        float(xy["x_min_um"]),
        float(xy["y_min_um"]),
        float(geometry["z_min_um"]),
        float(xy["x_max_um"]),
        float(xy["y_max_um"]),
        float(geometry["z_max_um"]),
    )


def model_materials(
    sidecar: Mapping[str, Any], stack: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """The stack's record of each material a volume of the model names; a volume naming none is refused."""
    catalog = stack.get("materials", {})
    names = sorted({volume["material"] for volume in sidecar["volumes"].values()})
    missing = [name for name in names if not isinstance(catalog.get(name), Mapping)]
    if missing:
        raise ValueError(f"the stack has no material record for {missing}")
    return {name: dict(catalog[name]) for name in names}


def check_region_box(sidecar: Mapping[str, Any], bounds: Sequence[float]) -> None:
    """Refuse a model whose own box is not the ``VACUUM_REGION`` box AEDT's Region will take.

    Palace's outer boundary is the box of the SGB model; AEDT's is its Region.  The two agree only when
    every vertex of the model lies in the vacuum box and the box's corners are reached, so the box of
    the model's vertices must equal ``bounds`` within the sidecar's tolerance.
    """
    vertices = sidecar["expected"]["vertices_um"]
    observed = [min(v[k] for v in vertices) for k in range(3)] + [
        max(v[k] for v in vertices) for k in range(3)
    ]
    tolerance = float(sidecar["tolerance_um"])
    if any(abs(a - b) > tolerance for a, b in zip(observed, bounds, strict=True)):
        raise ValueError(
            f"the SGB model spans {observed} um, not the VACUUM_REGION box {list(bounds)} um "
            "that AEDT's Region takes"
        )


def bind_junctions(
    effective_input: GeometryBuildInput, junctions: Sequence[PlanarJunction]
) -> tuple[SgbJunction, ...]:
    """Bind each junction to the port sheet its plan's ``RectangularJunctionSpec`` derived.

    The junction's polygon and its two Nets must be the ones the spec names; a junction without a spec
    is refused rather than partitioned again here.
    """
    by_junction = {
        region.metadata.get("rectangular_junction_partition"): region
        for region in effective_input.port_sheet_regions
        if region.metadata.get("rectangular_junction_partition") is not None
    }
    out = []
    for junction in junctions:
        if not isinstance(junction, PlanarJunction):
            raise TypeError("junctions must contain PlanarJunction records")
        region = by_junction.get(junction.junction_id)
        if region is None:
            raise ValueError(
                f"junction {junction.junction_id!r} has no RectangularJunctionSpec in the plan; "
                "declare it with GeometryPlan.set_rectangular_junctions"
            )
        binding = region.metadata["rectangular_junction_binding"]
        declared = (
            binding["source_polygon_id"],
            binding["terminal_a"]["net_id"],
            binding["terminal_b"]["net_id"],
        )
        requested = (
            junction.source_polygon_id,
            junction.terminal_a_net,
            junction.terminal_b_net,
        )
        if declared != requested:
            raise ValueError(
                f"junction {junction.junction_id!r} names polygon and Nets {requested}, "
                f"its RectangularJunctionSpec {declared}"
            )
        out.append(SgbJunction(junction, f"PORT::{region.port_sheet_id}"))
    if len({item.junction.junction_id for item in out}) != len(out):
        raise ValueError("junction ids must be unique")
    return tuple(out)


def _check_junction_ports(
    sidecar: Mapping[str, Any], junctions: Sequence[SgbJunction]
) -> None:
    groups = {port["group"] for port in sidecar["ports"]}
    missing = [item.port_group for item in junctions if item.port_group not in groups]
    if missing:
        raise ValueError(f"the SGB model has no lumped-port group {missing}")
