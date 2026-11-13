"""The Eigenmode problem: lumped ports, the EPR selection, the mode controls and the Eigenmode config.

Everything else (plan, vacuum, Surface EPR, mesh controls, ``mesh()``, ``simulation_drc``, the summaries and
the handoff) is ``PalaceSimulation`` in ``core``; this file shows only what the Eigenmode problem adds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from scgsim.geometry.source.plan import GeometryPlanSnapshot

from ..config.config import build_eigenmode_config
from ..config.config_models import EigenmodeConfigBuildResult, EigenmodeSettings
from ..config.index import EprRequest, check_epr_request
from ..config.numerical import NumericalControls, ProblemDefaults
from ..source.ports import RequestedPort
from ..source.transforms import validate_nonempty_string, validate_positive_number
from .core import PalaceSimulation, index_map

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
    epr_request: EprRequest | None = None
    num_modes: int = 10
    target_hz: float | None = None
    eigenmode_tolerance: float = 1e-6
    save_fields: int = 0

    problem = "Eigenmode"
    # Nonconformal AMR, with the pass limit and its tolerance left for the user to choose.
    numerical_defaults: ClassVar[ProblemDefaults] = ProblemDefaults(
        amr_max_passes=None, amr_nonconformal=True, amr_tolerance=None
    )
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
        """Select reported EPR quantities without changing the physical solve; stored as an ``EprRequest``.

        ``write_config()`` refuses a selection that names a binding the config does not have.

        Raises:
            TypeError: a selection is not a sequence of IDs.
            ValueError: a selection holds an empty or repeated ID, or an interface other than MA, MS, SA.
        """
        self.epr_request = EprRequest.selected(
            surface_interfaces=surface_interfaces,
            bulk_domain_ids=bulk_domain_ids,
            port_names=port_names,
        )
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

        All values are checked (``EigenmodeSettings``) before any is stored.

        Raises:
            ValueError: a value is not positive (``save``: negative).
            TypeError: ``save`` is not an integer.
        """
        settings = EigenmodeSettings(
            num_modes=num_modes, target_hz=target, tolerance=tolerance, save_fields=save
        )
        self.num_modes = settings.num_modes
        self.target_hz = settings.target_hz
        self.eigenmode_tolerance = settings.tolerance
        self.save_fields = settings.save_fields
        self._fall_back("set_eigenmode")

    def set_numerical(  # noqa: PLR0913 -- the public keyword controls of set_numerical
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
        """Set solver, FEM, AMR, and output controls for the next config write; stored as ``NumericalControls``.

        ``amr_max_passes`` is Palace ``MaxIts``. ``amr_nonconformal`` is Palace
        ``Model.Refinement.Nonconformal``. Omitted AMR controls retain their
        settings. New Eigenmode requests use NC and require explicit case
        controls before config generation; max passes zero disables AMR.
        ``amr_nc_levels`` sets native ``MaxNCLevels``; zero means no limit.
        Omitted Krylov and nonconformal-level controls retain current choices.
        Explicit ``ksp_type="Default"`` uses Palace's problem-specific default.

        How it differs from ``ElectrostaticSim.set_numerical``: here an omitted ``amr_max_passes``,
        ``amr_nonconformal``, ``amr_tolerance`` or ``amr_update_fraction`` keeps its current value; there each
        takes its default (0, False, 0.01, None).

        Raises:
            ValueError: a control is out of its range or choices.
            TypeError: a control has the wrong type.
        """
        current = self.numerical
        self._set_numerical(
            NumericalControls(
                order=order,
                tolerance=tolerance,
                max_iterations=max_iterations,
                solver_type=solver_type,
                preconditioner=preconditioner,
                device=device,
                ksp_type=current.ksp_type if ksp_type is None else ksp_type,
                amr_max_passes=_kept(amr_max_passes, current.amr_max_passes),
                amr_nonconformal=_kept(amr_nonconformal, current.amr_nonconformal),
                amr_nc_levels=_kept(amr_nc_levels, current.amr_nc_levels),
                amr_tolerance=_kept(amr_tolerance, current.amr_tolerance),
                amr_update_fraction=_kept(
                    amr_update_fraction, current.amr_update_fraction
                ),
                save_adapt_iterations=save_adapt_iterations,
                save_adapt_mesh=save_adapt_mesh,
                estimator_mg=estimator_mg,
                output_paraview=output_paraview,
                output_grid_function=output_grid_function,
            )
        )

    def write_config(self) -> Path:
        """Write ``config.json`` and its metadata for the current mesh, ports and Eigenmode settings.

        Returns:
            The path of the written ``config.json``.

        Raises:
            ValueError: ``mesh()`` has not built the current mesh, AMR is left open, the config refuses the
                bindings, or the EPR request names a binding the config does not have.
            FileNotFoundError: a file of the current mesh is gone from the run folder.
        """
        mesh, common = self._start_config()
        result = build_eigenmode_config(
            common, ports=self._resolved_ports, eigenmode=self._eigenmode_settings()
        )
        check_epr_request(self.epr_request, result.index_entries)
        return self._finish_config(mesh, result.config, self._metadata(result))

    def _eigenmode_settings(self) -> EigenmodeSettings:
        """The mode controls as the config reads them, checked again in case an attribute was assigned."""
        return EigenmodeSettings(
            num_modes=self.num_modes,
            target_hz=self.target_hz,
            tolerance=self.eigenmode_tolerance,
            save_fields=self.save_fields,
        )

    def _metadata(
        self, result: EigenmodeConfigBuildResult
    ) -> tuple[tuple[str, dict[str, Any]], ...]:
        """The ``metadata/`` files written beside ``config.json``, in the order they are written."""
        extra = (
            {}
            if self.epr_request is None
            else {"epr_request": self.epr_request.to_payload()}
        )
        return (
            ("palace_index_map.json", index_map(result.index_entries, **extra)),
            (
                "palace_material_resolution.json",
                {"schema_version": 1, "solution_volumes": result.material_resolution},
            ),
            (
                "port_information.json",
                {"schema_version": 1, "ports": result.port_information},
            ),
            ("palace_numerical_controls.json", self.numerical.to_payload()),
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
            "epr_request": None
            if self.epr_request is None
            else asdict(self.epr_request),
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


def _kept[Value](given: Value | None, current: Value) -> Value:
    """``given``, or the current setting when the control was omitted (None)."""
    return current if given is None else given


__all__ = ["EigenmodeSim"]
