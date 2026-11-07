"""Layout-time simulation check: report every geometry finding of a component before any mesh or solve.

``simulation_drc`` is the entry. It owns the rule order and the report shapes; the geometry itself
stays with the production stages it calls. Rules: 1 ``polygon-validity``, 2 ``semantic-contract``,
3 ``physical-interpretation``, 4 ``sgb-structure``, 5 ``surface-epr-bands``.
"""

from scgsim.geometry.drc.check import simulation_drc
from scgsim.geometry.drc.report import (
    LayoutCheckError,
    LayoutCheckReport,
    LayoutFinding,
    SkippedRule,
)

__all__ = [
    "LayoutCheckError",
    "LayoutCheckReport",
    "LayoutFinding",
    "SkippedRule",
    "simulation_drc",
]
