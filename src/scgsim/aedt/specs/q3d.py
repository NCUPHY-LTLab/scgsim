"""The Q3D request on an SGB model: its Nets, optional source and sink, and the matrix setup.

The request carries the :class:`PreparedSgbModel` AEDT imports (issue #67), as HFSS Eigenmode does.  Its
payload is a handoff payload: the model lives in the handoff's ``model/`` folder, so a payload is decoded
against the directory that holds it (``base_dir``) and every model file is checked again there.  A payload of
the native builder (one that declares ``bodies``) is refused with the reason (issue #67 decision 6).

The Nets are the model's: one :class:`Q3dNetSpec` per Net of the model, named by its AEDT name (the
sanitised Net id), over exactly the model's conductor objects of that Net
(:func:`scgsim.aedt.preparation.sgb_q3d.q3d_net_objects`), with the request's Q3D type.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Literal, cast, get_args

from scgsim.aedt.preparation.sgb_handoff import bind_sgb_model, model_payload
from scgsim.aedt.preparation.sgb_model import PreparedSgbModel
from scgsim.aedt.preparation.sgb_q3d import q3d_net_objects
from scgsim.aedt.specs.common import (
    LOCKED_PYAEDT,
    OFFICIAL_PYAEDT_SOURCE_URL,
    Q3D_SCHEMA_VERSION,
    MatrixRunControl,
    Side,
    project_filename,
    project_name_from_payload,
    text_value,
)
from scgsim.aedt.specs.modeling import Modeling
from scgsim.aedt.versions import normalize_aedt_version

NATIVE_PAYLOAD_REFUSAL = (
    "this Q3D request declares natively built bodies; AEDT runs and reads only SGB models since "
    "issue #67, prepare it again"
)


@dataclass(frozen=True)
class Q3dTerminalPair:
    """Caller-selected global terminal sides on exact source Entity identities.

    Each selected Entity must be exactly one conductor object of its Net in the SGB model.
    """

    source_entity_id: str
    source_side: Side
    sink_entity_id: str
    sink_side: Side

    def __post_init__(self) -> None:
        for name in ("source_entity_id", "sink_entity_id"):
            object.__setattr__(self, name, text_value(getattr(self, name), name))
        if self.source_side not in get_args(Side) or self.sink_side not in get_args(
            Side
        ):
            raise ValueError("Q3D source and sink sides are invalid")


@dataclass(frozen=True)
class Q3dNetSpec:
    """One exact connected Q3D net and optional signal source/sink."""

    name: str
    net_type: Literal["Signal", "Ground"]
    object_names: tuple[str, ...]
    source_object: str | None = None
    source_side: Side | None = None
    sink_object: str | None = None
    sink_side: Side | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", text_value(self.name, "net.name"))
        if self.net_type not in {"Signal", "Ground"}:
            raise ValueError("Q3D net_type must be Signal or Ground")
        objects = tuple(
            text_value(value, "net.object_name") for value in self.object_names
        )
        if not objects or len(set(objects)) != len(objects):
            raise ValueError("Q3D net object_names must be nonempty and unique")
        object.__setattr__(self, "object_names", objects)
        terminal_values = (
            self.source_object,
            self.source_side,
            self.sink_object,
            self.sink_side,
        )
        if self.net_type == "Ground":
            if any(value is not None for value in terminal_values):
                raise ValueError("Q3D Ground nets must not define source or sink")
            return
        if all(value is None for value in terminal_values):
            return
        if any(value is None for value in terminal_values):
            raise ValueError(
                "Q3D Signal terminals must be all absent or a complete source/sink pair"
            )
        source = text_value(self.source_object, "net.source_object")
        sink = text_value(self.sink_object, "net.sink_object")
        if source not in objects or sink not in objects:
            raise ValueError("Q3D source and sink objects must belong to their net")
        if self.source_side not in get_args(Side) or self.sink_side not in get_args(
            Side
        ):
            raise ValueError("Q3D source and sink sides are invalid")
        object.__setattr__(self, "source_object", source)
        object.__setattr__(self, "sink_object", sink)

    @property
    def has_terminals(self) -> bool:
        """Whether this Net declares a source and a sink."""
        return self.source_object is not None

    def to_payload(self) -> dict[str, Any]:
        """Return this request record in its canonical payload shape."""
        return {
            "name": self.name,
            "net_type": self.net_type,
            "object_names": list(self.object_names),
            "source_object": self.source_object,
            "source_side": self.source_side,
            "sink_object": self.sink_object,
            "sink_side": self.sink_side,
        }


@dataclass(frozen=True)
class Q3dSpec:
    """One Q3D model of an SGB model with optional matrix setup.

    Explicit run_control=None declares model preparation without solve setup.  The Region is the model's
    ``VACUUM_REGION`` box (issue #67 decision 5).  ``grounded_region_net`` names a new Ground Net made of the
    Region's six faces, for a capacitance-only request.
    """

    modeling: Modeling
    project_name: str
    design_name: str
    model: PreparedSgbModel
    nets: tuple[Q3dNetSpec, ...]
    run_control: MatrixRunControl | None
    solve_ac_rl: bool = True
    grounded_region_net: str | None = None
    aedt_version: str = dataclass_field(kw_only=True)
    pyaedt_version: str = LOCKED_PYAEDT
    schema_version: Literal["scgsim.aedt.q3d.v3"] = Q3D_SCHEMA_VERSION

    @property
    def mode(self) -> Literal["q3d"]:
        """Return the solver family represented by this request."""
        return "q3d"

    def __post_init__(self) -> None:
        if self.schema_version != Q3D_SCHEMA_VERSION:
            raise ValueError("unsupported Q3D schema")
        object.__setattr__(
            self, "aedt_version", normalize_aedt_version(self.aedt_version)
        )
        if self.pyaedt_version != LOCKED_PYAEDT:
            raise ValueError("V1 requires PyAEDT 1.3.0")
        project = project_filename(self.project_name, "project_name")
        object.__setattr__(self, "project_name", project.stem)
        object.__setattr__(
            self, "design_name", text_value(self.design_name, "design_name")
        )
        if not isinstance(self.model, PreparedSgbModel):
            raise TypeError("model must be PreparedSgbModel")
        if self.model.modeling != self.modeling:
            raise ValueError("request and prepared model modeling differ")
        if self.run_control is not None and not isinstance(
            self.run_control, MatrixRunControl
        ):
            raise TypeError("run_control must be MatrixRunControl or None")
        nets = tuple(self.nets)
        _check_model_nets(self.model, nets)
        _check_terminals(self.solve_ac_rl, nets)
        _check_grounded_region(self.grounded_region_net, self.solve_ac_rl, nets)
        object.__setattr__(self, "nets", nets)

    def to_payload(self) -> dict[str, Any]:
        """Return this request record in its canonical payload shape; its model names ``model/``."""
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "modeling": self.modeling,
            "aedt": {"requested_version": self.aedt_version},
            "pyaedt": {
                "locked_version": self.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            "project": {"name": self.project_name, "design": self.design_name},
            "model": model_payload(self.model),
            "nets": [net.to_payload() for net in self.nets],
            "run_control": None
            if self.run_control is None
            else self.run_control.to_payload(),
            "solve_ac_rl": self.solve_ac_rl,
            "grounded_region_net": self.grounded_region_net,
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> Q3dSpec:
        """Decode this request from its payload, binding its model under ``base_dir``."""
        if payload.get("schema_version") != Q3D_SCHEMA_VERSION:
            raise ValueError("unsupported Q3D schema")
        if {"bodies", "geometry_source", "region_padding_um"} & set(payload):
            raise ValueError(NATIVE_PAYLOAD_REFUSAL)
        expected = {
            "schema_version",
            "mode",
            "modeling",
            "aedt",
            "pyaedt",
            "project",
            "model",
            "nets",
            "run_control",
            "solve_ac_rl",
            "grounded_region_net",
        }
        if set(payload) != expected or payload.get("mode") != "q3d":
            raise ValueError("Q3D payload members are not canonical")
        if base_dir is None:
            raise ValueError(
                "a Q3D request is read against the handoff directory that holds its model"
            )
        project = payload["project"]
        return cls(
            modeling=cast(Modeling, payload["modeling"]),
            project_name=project_name_from_payload(project.get("name")),
            design_name=text_value(project.get("design"), "project.design"),
            model=bind_sgb_model(payload["model"] or {}, Path(base_dir)),
            nets=tuple(Q3dNetSpec(**net) for net in payload["nets"]),
            run_control=(
                None
                if payload["run_control"] is None
                else MatrixRunControl(**payload["run_control"])
            ),
            solve_ac_rl=payload["solve_ac_rl"],
            grounded_region_net=payload["grounded_region_net"],
            aedt_version=payload["aedt"]["requested_version"],
            pyaedt_version=payload["pyaedt"]["locked_version"],
        )


def _check_model_nets(model: PreparedSgbModel, nets: tuple[Q3dNetSpec, ...]) -> None:
    """The request's Nets must be the model's, in its order, each over exactly its conductor objects."""
    if any(not isinstance(net, Q3dNetSpec) for net in nets):
        raise TypeError("Q3D nets must contain Q3dNetSpec records")
    expected = q3d_net_objects(model)
    if [net.name for net in nets] != list(expected):
        raise ValueError(
            f"Q3D nets {[net.name for net in nets]} are not the model's Nets {list(expected)}"
        )
    for net in nets:
        if net.object_names != expected[net.name]:
            raise ValueError(
                f"Q3D net {net.name!r} names objects {list(net.object_names)}, the model "
                f"{list(expected[net.name])}"
            )
    if not any(net.net_type == "Signal" for net in nets):
        raise ValueError("Q3D requires at least one Signal net")


def _check_terminals(solve_ac_rl: bool, nets: tuple[Q3dNetSpec, ...]) -> None:
    if not isinstance(solve_ac_rl, bool):
        raise TypeError("solve_ac_rl must be boolean")
    if solve_ac_rl and any(
        net.net_type == "Signal" and not net.has_terminals for net in nets
    ):
        raise ValueError("Q3D AC/RL requires complete terminals for every Signal net")


def _check_grounded_region(
    grounded_region_net: str | None,
    solve_ac_rl: bool,
    nets: tuple[Q3dNetSpec, ...],
) -> None:
    if grounded_region_net is None:
        return
    text_value(grounded_region_net, "grounded_region_net")
    if solve_ac_rl:
        raise ValueError("grounded_region_net requires solve_ac_rl=False")
    if grounded_region_net in {net.name for net in nets}:
        raise ValueError("grounded_region_net must name a new dedicated enclosure net")
