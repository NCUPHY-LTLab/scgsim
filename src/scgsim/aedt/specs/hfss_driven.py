"""The HFSS Driven Terminal request on an SGB model: two lumped ports, an optional setup and sweep.

The request carries the :class:`PreparedSgbModel` AEDT imports (issue #67).  Its ports are the model's
``LumpedTerminalPort`` records, each on the ``PORT::<port_sheet_id>`` group it names
(``prepare_sgb_model(..., ports=...)``).  Driven wave ports and Driven Modal ports are refused when the model
is prepared (decision 8, follow-up NCUPHY-LTLab/scgsim#135).  A request payload is a handoff payload: its
model lives in the handoff's ``model/`` folder, so it is decoded against the directory that holds it
(``base_dir``).  A payload of the native builder (one that carries ``geometry``) is refused with the reason
(decision 6).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from scgsim.aedt.preparation.sgb_handoff import bind_sgb_model, model_payload
from scgsim.aedt.preparation.sgb_model import PreparedSgbModel, port_group
from scgsim.aedt.specs.common import (
    DRIVEN_GEOMETRY_SCHEMA_VERSION,
    LOCKED_PYAEDT,
    OFFICIAL_PYAEDT_SOURCE_URL,
    LumpedTerminalPort,
    project_filename,
    project_name_from_payload,
    text_value,
)
from scgsim.aedt.specs.hfss_common import FrequencySweepSpec, HfssRunControl
from scgsim.aedt.specs.modeling import Modeling
from scgsim.aedt.versions import normalize_aedt_version
from scgsim.contracts.sgb_sidecar import read_sidecar

NATIVE_DRIVEN_REFUSAL = (
    "this HFSS Driven request carries a natively built geometry; AEDT runs and reads only SGB models "
    "since issue #67, prepare it again"
)
_PAYLOAD_KEYS = {
    "schema_version",
    "mode",
    "modeling",
    "aedt",
    "pyaedt",
    "project",
    "model",
    "run_control",
}


@dataclass(frozen=True)
class HfssDrivenGeometrySpec:
    """One HFSS Driven Terminal model of an SGB model, its two lumped ports and an optional setup.

    An explicit null ``run_control`` declares model-only preparation.
    """

    modeling: Modeling
    project_name: str
    design_name: str
    model: PreparedSgbModel
    run_control: HfssRunControl | None = None
    aedt_version: str = field(kw_only=True)
    pyaedt_version: str = LOCKED_PYAEDT

    @property
    def mode(self) -> Literal["terminal"]:
        """Return the solver mode declared by this request."""
        return "terminal"

    @property
    def ports(self) -> tuple[LumpedTerminalPort, ...]:
        """The model's lumped Terminal ports, in port order."""
        return self.model.lumped_ports

    def __post_init__(self) -> None:
        if not isinstance(self.model, PreparedSgbModel):
            raise TypeError("model must be PreparedSgbModel")
        if self.model.modeling != self.modeling:
            raise ValueError("request and prepared model modeling differ")
        if (
            self.model.junctions
            or self.model.lumped_rlcs
            or self.model.surface_epr_specs is not None
        ):
            raise ValueError(
                "a Driven model carries no junctions, lumped RLCs or Surface-EPR specs; "
                "they belong to HFSS Eigenmode"
            )
        object.__setattr__(
            self, "aedt_version", normalize_aedt_version(self.aedt_version)
        )
        if self.pyaedt_version != LOCKED_PYAEDT:
            raise ValueError("V1 requires PyAEDT 1.3.0")
        object.__setattr__(
            self,
            "project_name",
            project_filename(self.project_name, "project_name").stem,
        )
        object.__setattr__(
            self, "design_name", text_value(self.design_name, "design_name")
        )
        if self.run_control is not None and not isinstance(
            self.run_control, HfssRunControl
        ):
            raise TypeError("run_control must be HfssRunControl or None")
        validate_driven_ports(self.model)

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical payload; its model names the handoff's ``model/`` folder."""
        return {
            "schema_version": DRIVEN_GEOMETRY_SCHEMA_VERSION,
            "mode": self.mode,
            "modeling": self.modeling,
            "aedt": {"requested_version": self.aedt_version},
            "pyaedt": {
                "locked_version": self.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            "project": {"name": self.project_name, "design": self.design_name},
            "model": model_payload(self.model),
            "run_control": None
            if self.run_control is None
            else self.run_control.to_payload(),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, base_dir: Path | None
    ) -> HfssDrivenGeometrySpec:
        """Decode the canonical payload, binding its model under ``base_dir``."""
        if (
            payload.get("schema_version") != DRIVEN_GEOMETRY_SCHEMA_VERSION
            or payload.get("mode") != "terminal"
        ):
            raise ValueError("unsupported HFSS Driven Terminal schema")
        if "geometry" in payload:
            raise ValueError(NATIVE_DRIVEN_REFUSAL)
        if set(payload) != _PAYLOAD_KEYS:
            raise ValueError("HFSS Driven Terminal payload members are not canonical")
        if base_dir is None:
            raise ValueError(
                "an HFSS Driven request is read against the handoff directory that holds its model"
            )
        run = payload["run_control"]
        if run is not None:
            run = dict(run)
            run["sweep"] = FrequencySweepSpec(**run["sweep"])
        project = payload["project"]
        return cls(
            modeling=cast(Modeling, payload["modeling"]),
            project_name=project_name_from_payload(project.get("name")),
            design_name=text_value(project.get("design"), "project.design"),
            model=bind_sgb_model(payload["model"] or {}, Path(base_dir)),
            run_control=None if run is None else HfssRunControl(**run),
            aedt_version=payload["aedt"]["requested_version"],
            pyaedt_version=payload["pyaedt"]["locked_version"],
        )


def validate_driven_ports(model: PreparedSgbModel) -> None:
    """Check the model's lumped ports against the two-port V1 contract and the sidecar's port sheets.

    The ports are 1 and 2 with distinct names and one shared, ordered reference.  No Entity of a port is both
    signal and reference.  Each port sheet joins two host conductors; one must be a signal Entity of its port
    and the other a reference Entity.
    """
    ports = model.lumped_ports
    if (
        tuple(port.index for port in ports) != (1, 2)
        or len({port.name for port in ports}) != 2
    ):
        raise ValueError(
            "HFSS Driven Terminal V1 requires two lumped ports, 1 and 2, with distinct names"
        )
    if ports[0].reference_entity_ids != ports[1].reference_entity_ids:
        raise ValueError(
            f"Terminal ports {ports[0].name!r} and {ports[1].name!r} must share one ordered reference"
        )
    sidecar = read_sidecar(model.path("sidecar"))
    hosts = {
        record["group"]: tuple(record["host_entity_ids"])
        for record in sidecar["port"].values()
    }
    for port in ports:
        both = sorted(set(port.signal_entity_ids) & set(port.reference_entity_ids))
        if both:
            raise ValueError(
                f"port {port.name!r}: {both} is both a signal and a reference Entity; a lumped port's "
                "Terminal goes on its signal conductor and the other conductor is its reference"
            )
        sheet_hosts = hosts[port_group(port.port_sheet_id)]
        signal = [h for h in sheet_hosts if h in port.signal_entity_ids]
        reference = [h for h in sheet_hosts if h in port.reference_entity_ids]
        if len(sheet_hosts) != 2 or len(signal) != 1 or len(reference) != 1:
            raise ValueError(
                f"port {port.name!r}: its sheet {port.port_sheet_id!r} joins {list(sheet_hosts)}; one must "
                f"be among its signal Entities {list(port.signal_entity_ids)} and the other among its "
                f"reference Entities {list(port.reference_entity_ids)}"
            )
