"""The stages of a Palace simulation, which call needs which stage, and how far each call falls back.

A simulation moves forward one stage per call: ``set_plan`` binds a plan, ``mesh()`` writes ``palace.msh``,
``write_config()`` writes ``config.json`` and ``prepare_handoff()`` writes the run script and archive.
Each stage holds the outputs of the stages before it.  A setter that changes what a stage was built from
falls back to the last stage it leaves valid, so the outputs it changes are never read stale:

=====================  =====================  ===========================================
Call                   Needs                  Falls back to before it runs or once it is set
=====================  =====================  ===========================================
``set_plan``           nothing                ``BOUND`` (a new plan; nothing built is kept)
geometry setters       nothing                ``BOUND``
``set_surface_epr``    ``BOUND``              ``BOUND`` (its specs are checked against the plan's materials)
problem setters        nothing                ``MESHED``
``input_summary``      ``BOUND``              nothing is discarded
``mesh``               ``BOUND``              ``BOUND``
``mesh_summary``       ``MESHED``             nothing is discarded
``write_config``       ``MESHED``             ``MESHED``
``prepare_handoff``    ``CONFIG_WRITTEN``     nothing is discarded
=====================  =====================  ===========================================

The geometry setters are the others whose value ``mesh()`` reads (output folder, vacuum, mesh controls,
Eigenmode ports); the problem setters are those only ``write_config()`` reads.  ``CALLS`` is
that table as code; the simulation core reads it and nothing else decides these rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal


class Stage(IntEnum):
    """How far a simulation has come; a later stage holds every output of the earlier ones."""

    UNBOUND = 0
    BOUND = 1  # set_plan(): the plan and its materials
    MESHED = 2  # mesh(): palace.msh, its physical groups and the resolved ports
    CONFIG_WRITTEN = 3  # write_config(): config.json and its metadata
    HANDED_OFF = 4  # prepare_handoff(): the run script, receipt recorder and archive


Call = Literal[
    "set_plan",
    "set_output_dir",
    "set_vacuum_region",
    "set_surface_epr",
    "set_mesh",
    "add_port",
    "set_epr_request",
    "set_eigenmode",
    "add_terminal",
    "add_ground",
    "set_electrostatic",
    "set_numerical",
    "input_summary",
    "mesh",
    "mesh_summary",
    "write_config",
    "prepare_handoff",
]


@dataclass(frozen=True)
class CallRule:
    """The stage one call needs, its refusal below that stage, and the stage it falls back to.

    ``falls_back_to`` is None for a call that reads only, so it discards nothing.
    """

    needs: Stage = Stage.UNBOUND
    falls_back_to: Stage | None = None
    refusal: str = ""
    error: type[Exception] = ValueError


_GEOMETRY_CHANGE = CallRule(falls_back_to=Stage.BOUND)
_PROBLEM_CHANGE = CallRule(falls_back_to=Stage.MESHED)

CALLS: dict[Call, CallRule] = {
    "set_plan": _GEOMETRY_CHANGE,
    "set_output_dir": _GEOMETRY_CHANGE,
    "set_vacuum_region": _GEOMETRY_CHANGE,
    "set_surface_epr": CallRule(
        needs=Stage.BOUND,
        falls_back_to=Stage.BOUND,
        # The specs name materials, and the bound plan's stack is what they are checked against.
        refusal="No plan materials to check the Surface EPR specs against: call set_plan() before set_surface_epr().",
    ),
    "set_mesh": _GEOMETRY_CHANGE,
    "add_port": _GEOMETRY_CHANGE,
    "set_epr_request": _PROBLEM_CHANGE,
    "set_eigenmode": _PROBLEM_CHANGE,
    "add_terminal": _PROBLEM_CHANGE,
    "add_ground": _PROBLEM_CHANGE,
    "set_electrostatic": _PROBLEM_CHANGE,
    "set_numerical": _PROBLEM_CHANGE,
    "input_summary": CallRule(
        needs=Stage.BOUND,
        refusal="No plan to summarise: call set_plan() before input_summary().",
    ),
    "mesh": CallRule(
        needs=Stage.BOUND,
        falls_back_to=Stage.BOUND,
        refusal="No plan to mesh: call set_plan() before mesh().",
    ),
    "mesh_summary": CallRule(
        needs=Stage.MESHED,
        refusal="No mesh to summarise: call mesh() before mesh_summary().",
        error=RuntimeError,
    ),
    "write_config": CallRule(
        needs=Stage.MESHED,
        falls_back_to=Stage.MESHED,
        refusal="No mesh to write a config for: call set_plan() and mesh() before write_config().",
    ),
    "prepare_handoff": CallRule(
        needs=Stage.CONFIG_WRITTEN,
        refusal="No config to hand off: call mesh() and write_config() before prepare_handoff().",
    ),
}

__all__ = ["CALLS", "Call", "CallRule", "Stage"]
