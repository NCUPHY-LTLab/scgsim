"""The Eigenmode problem: lumped ports, the EPR selection, the mode controls and the Eigenmode config.

Everything else (plan, vacuum, Surface EPR, mesh controls, ``mesh()``, ``simulation_drc``, the summaries and
the handoff) is ``PalaceSimulation`` in ``core``; this file shows only what the Eigenmode problem adds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from scgsim.geometry.source.plan import GeometryPlanSnapshot

from ..config.config import build_eigenmode_config, configure_numerical_controls
from ..source.ports import RequestedPort
from ..source.transforms import (
    validate_non_negative_int,
    validate_nonempty_string,
    validate_positive_number,
)
from ..results.index_contract import INDEX_MAP_SCHEMA_VERSION
from .core import PalaceSimulation

# The geometry build never reads a port's inductance, but ``add_port`` requires a positive one.
_PORT_INDUCTANCE_H = 1.0e-9


@dataclass
class EigenmodeSim(PalaceSimulation):
    """Prepare an SGB-authored Eigenmode mesh, config, and manual handoff.

    The calls run in this order; each later call needs the earlier ones::

        sim = EigenmodeSim()
        sim.set_plan(plan.prepare())
        sim.set_output_dir("runs/xmon")
        sim.set_surface_epr(representation="B", specs=specs)  # optional for Eigenmode
        sim.add_port("chip/P1", layer="metal", layout_sheet=True, inductance=10e-9)
        sim.set_eigenmode(num_modes=2, target=4e9)
        sim.set_numerical(amr_max_passes=0)
        sim.mesh()
        sim.write_config()
        sim.prepare_handoff(
            profile="direct-local", executable="palace", resources={"processes": 1}
        )
    """

    ports: list[RequestedPort] = field(default_factory=list)
    epr_request: dict[str, tuple[str, ...] | None] | None = None
    num_modes: int = 10
    target_hz: float | None = None
    eigenmode_tolerance: float = 1e-6
    save_fields: int = 0
    numerical: dict[str, Any] = field(
        default_factory=lambda: configure_numerical_controls(
            amr_nonconformal=True,
            amr_max_passes=None,
            amr_tolerance=None,
            amr_update_fraction=None,
        )
    )

    problem = "Eigenmode"
    # Eigenmode may skip Surface EPR: mesh() then requests no bands.
    mesh_requires_surface_epr: ClassVar[bool] = False

    def add_port(
        self,
        name: str,
        *,
        layer: str,
        layout_sheet: bool = False,
        inductance: float,
    ) -> None:
        """Bind one authored GDSFactory port to its target logical metal.

        The n-th port added is Palace port ``P<n>``.  A port is geometry: it invalidates the mesh.

        Raises:
            ValueError: ``layout_sheet`` is not True, a name or layer is empty, the port is already
                added, or ``inductance`` is not positive.
        """
        if layout_sheet is not True:
            raise ValueError(
                "Eigenmode supports only authored sheet ports: pass layout_sheet=True."
            )
        name = validate_nonempty_string(name, "port name")
        layer = validate_nonempty_string(layer, "port target layer")
        if any(port.name == name for port in self.ports):
            raise ValueError(f"layout_sheet port {name!r} is already added.")
        self.ports.append(
            RequestedPort(
                name=name,
                layer=layer,
                circuit={"L": validate_positive_number(inductance, "inductance")},
            )
        )
        self._fall_back("add_port")

    def set_epr_request(
        self,
        *,
        surface_interfaces: Sequence[str] | None = None,
        bulk_domain_ids: Sequence[str] | None = None,
        port_names: Sequence[str] | None = None,
    ) -> None:
        """Select reported EPR quantities without changing the physical solve.

        ``write_config()`` refuses a selection that names a binding the config does not have.

        Raises:
            TypeError: a selection is not a sequence of IDs.
            ValueError: a selection holds an empty or repeated ID, or an interface other than MA, MS, SA.
        """
        surfaces = _selected(surface_interfaces, "surface_interfaces")
        if surfaces is not None and set(surfaces) - {"MA", "MS", "SA"}:
            raise ValueError("surface_interfaces supports only MA, MS, and SA")
        self.epr_request = {
            "surface_interfaces": surfaces,
            "bulk_domain_ids": _selected(bulk_domain_ids, "bulk_domain_ids"),
            "port_names": _selected(port_names, "port_names"),
        }
        self._fall_back("set_epr_request")

    def set_eigenmode(
        self,
        *,
        num_modes: int = 10,
        target: float | None = None,
        tolerance: float = 1e-6,
        save: int = 0,
    ) -> None:
        """Set the number of modes, the target frequency (Hz), the solver tolerance and the saved fields.

        All values are checked before any is stored.

        Raises:
            ValueError: a value is not positive (``save``: not a non-negative integer).
        """
        if (
            not isinstance(num_modes, int)
            or isinstance(num_modes, bool)
            or num_modes < 1
        ):
            raise ValueError("num_modes must be a positive integer.")
        if target is not None:
            target = validate_positive_number(target, "target")
        normalized_tolerance = validate_positive_number(tolerance, "tolerance")
        normalized_save = validate_non_negative_int(save, "save")
        self.num_modes = num_modes
        self.target_hz = target
        self.eigenmode_tolerance = normalized_tolerance
        self.save_fields = normalized_save
        self._fall_back("set_eigenmode")

    def set_numerical(  # noqa: PLR0913 - preserve the public keyword controls.
        self,
        *,
        order: int = 1,
        tolerance: float = 1e-6,
        max_iterations: int = 400,
        solver_type: str = "Default",
        preconditioner: str = "Default",
        device: str = "CPU",
        ksp_type: str | None = None,
        amr_max_passes: int | None = None,
        amr_nonconformal: bool | None = None,
        amr_nc_levels: int | None = None,
        amr_tolerance: float | None = None,
        amr_update_fraction: float | None = None,
        save_adapt_iterations: bool | None = None,
        save_adapt_mesh: bool | None = None,
        estimator_mg: bool | None = None,
        output_paraview: bool | None = None,
        output_grid_function: bool | None = None,
    ) -> None:
        """Set solver, FEM, AMR, and output controls for the next config write.

        ``amr_max_passes`` is Palace ``MaxIts``. ``amr_nonconformal`` is Palace
        ``Model.Refinement.Nonconformal``. Omitted AMR controls retain their
        settings. New Eigenmode requests use NC and require explicit case
        controls before config generation; max passes zero disables AMR.
        ``amr_nc_levels`` sets native ``MaxNCLevels``; zero means no limit.
        Omitted Krylov and nonconformal-level controls retain current choices.
        Explicit ``ksp_type="Default"`` uses Palace's problem-specific default.
        """
        current = self.numerical
        self.numerical = configure_numerical_controls(
            order=order,
            tolerance=tolerance,
            max_iterations=max_iterations,
            solver_type=solver_type,
            preconditioner=preconditioner,
            device=device,
            ksp_type=current.get("ksp_type") if ksp_type is None else ksp_type,
            amr_nc_levels=(
                current.get("amr_nc_levels") if amr_nc_levels is None else amr_nc_levels
            ),
            amr_max_passes=(
                current["amr_max_passes"] if amr_max_passes is None else amr_max_passes
            ),
            amr_nonconformal=(
                current["amr_nonconformal"]
                if amr_nonconformal is None
                else amr_nonconformal
            ),
            amr_tolerance=(
                current["amr_tolerance"] if amr_tolerance is None else amr_tolerance
            ),
            amr_update_fraction=(
                current["amr_update_fraction"]
                if amr_update_fraction is None
                else amr_update_fraction
            ),
            save_adapt_iterations=save_adapt_iterations,
            save_adapt_mesh=save_adapt_mesh,
            estimator_mg=estimator_mg,
            output_paraview=output_paraview,
            output_grid_function=output_grid_function,
        )
        self._fall_back("set_numerical")

    def write_config(self) -> Path:
        """Write ``config.json`` and its metadata for the current mesh, ports and Eigenmode settings.

        Returns:
            The path of the written ``config.json``.

        Raises:
            ValueError: ``mesh()`` has not built the current mesh, the config refuses the bindings, or the
                EPR request names a binding the config does not have.
            FileNotFoundError: a file of the current mesh is gone from the run folder.
        """
        mesh, materials = self._start_config()
        result = build_eigenmode_config(
            groups=mesh.groups,
            ports=self._resolved_ports,
            materials=materials,
            surface_epr_specs=self.surface_epr_specs or {},
            numerical=self.numerical,
            num_modes=self.num_modes,
            target_hz=self.target_hz,
            eigenmode_tolerance=self.eigenmode_tolerance,
            save_fields=self.save_fields,
            mesh_path=mesh.mesh_path,
        )
        _check_epr_request(self.epr_request, result.index_entries)
        return self._finish_config(
            mesh,
            result.config,
            metadata=(
                (
                    "palace_index_map.json",
                    _index_map(result.index_entries, self.epr_request),
                ),
                (
                    "palace_material_resolution.json",
                    {
                        "schema_version": 1,
                        "solution_volumes": result.material_resolution,
                    },
                ),
                (
                    "port_information.json",
                    {"schema_version": 1, "ports": result.port_information},
                ),
                ("palace_numerical_controls.json", self.numerical),
            ),
        )

    # ---- The core's questions ------------------------------------------------------------------------

    def _lumped_ports(self) -> list[RequestedPort]:
        """Every port added, in the order ``add_port`` received them."""
        return self.ports

    def _problem_summary(self) -> dict[str, Any]:
        """The ports, the EPR selection and the mode controls."""
        return {
            "ports": [
                {
                    "name": item.name,
                    "layer": item.layer,
                    "inductance_h": item.circuit["L"],
                }
                for item in self.ports
            ],
            "epr_request": self.epr_request,
            "num_modes": self.num_modes,
            "target_hz": self.target_hz,
            "eigenmode_tolerance": self.eigenmode_tolerance,
            "save_fields": self.save_fields,
        }

    def _add_drc_inputs(self, snapshot: GeometryPlanSnapshot) -> None:
        """Add every authored port of ``snapshot``, with a fixed inductance the geometry never reads."""
        for port in snapshot.stack["metadata"].get("port_sheet_source_layers", ()):
            self.add_port(
                port["name"],
                layer=port["target_layer"],
                layout_sheet=True,
                inductance=_PORT_INDUCTANCE_H,
            )


def _selected(values: Sequence[str] | None, name: str) -> tuple[str, ...] | None:
    """One ``set_epr_request`` selection as a tuple of distinct non-empty IDs, or None for all."""
    if values is None:
        return None
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of IDs or None")
    normalized = tuple(values)
    if any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError(f"{name} must contain non-empty text IDs")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must not repeat IDs")
    return normalized


def _check_epr_request(
    request: Mapping[str, tuple[str, ...] | None] | None,
    index_entries: Sequence[Mapping[str, Any]],
) -> None:
    """Refuse an EPR request that selects a binding the index of the config being written does not list."""
    if request is None:
        return
    available = {
        "surface_interfaces": {
            row["interface_type"]
            for item in index_entries
            if item["section"] == "Boundaries.Postprocessing.Dielectric"
            for row in item["metadata"]["rows"]
        },
        "bulk_domain_ids": {
            item["entry_name"]
            for item in index_entries
            if item["section"] == "Domains.Postprocessing.Energy"
        },
        "port_names": {
            item["port_name"]
            for item in index_entries
            if item["section"] == "Boundaries.LumpedPort"
        },
    }
    for name, configured in available.items():
        chosen = request[name]
        if chosen is not None and set(chosen) - configured:
            unknown = sorted(set(chosen) - configured)
            raise ValueError(
                f"EPR request selects an unknown {name} binding {unknown}; the config has "
                f"{sorted(configured)}: call set_epr_request() with those, then write_config()."
            )


def _index_map(
    index_entries: Sequence[Mapping[str, Any]],
    request: Mapping[str, tuple[str, ...] | None] | None,
) -> dict[str, Any]:
    """The ``palace_index_map.json`` payload: the index entries, and the EPR selection when one is set."""
    payload: dict[str, Any] = {
        "schema_version": INDEX_MAP_SCHEMA_VERSION,
        "entries": index_entries,
    }
    if request is not None:
        payload["epr_request"] = {
            "schema_version": "scgsim.palace.epr-selection.v1",
            **{
                name: None if values is None else list(values)
                for name, values in request.items()
            },
        }
    return payload


__all__ = ["EigenmodeSim"]
