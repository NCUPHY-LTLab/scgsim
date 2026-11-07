"""The result of a layout check: findings, skipped rules and their plain-text rendering.

Owns the shapes a layout repository reads from :func:`scgsim.geometry.drc.simulation_drc`. It knows
nothing about how a rule finds a defect.
"""

from __future__ import annotations

from dataclasses import dataclass

POLYGON_VALIDITY = "polygon-validity"  # rule 1
SEMANTIC_CONTRACT = "semantic-contract"  # rule 2
PHYSICAL_INTERPRETATION = "physical-interpretation"  # rule 3
SGB_STRUCTURE = "sgb-structure"  # rule 4
SURFACE_EPR_BANDS = "surface-epr-bands"  # rule 5

Layer = tuple[int, int]
"""A GDS layer and datatype."""
Location = tuple[float, float]
"""An ``(x, y)`` position in micrometres, in the frame of the component that was checked."""


@dataclass(frozen=True)
class LayoutFinding:
    """One defect: which rule found it, where it is and what to change."""

    rule: str
    layer: Layer | None
    entity: (
        str | None
    )  # a qualified semantic Entity ID, or the cell when no Entity exists yet
    location_um: Location | None
    message: str
    hint: str

    def render(self) -> str:
        """The finding as the text of a terminal or CI log."""
        where = [f"layer {self.layer[0]}/{self.layer[1]}"] if self.layer else []
        if self.entity:
            where.append(f"entity {self.entity}")
        if self.location_um:
            where.append(
                f"at ({self.location_um[0]:.4f}, {self.location_um[1]:.4f}) um"
            )
        place = ", ".join(where) or "no location"
        return f"[{self.rule}] {place}\n    {self.message}\n    fix: {self.hint}"


@dataclass(frozen=True)
class SkippedRule:
    """A rule that did not run, and why; a skipped rule is never a passed rule."""

    rule: str
    reason: str

    def render(self) -> str:
        """The skipped rule as the text of a terminal or CI log."""
        return f"[{self.rule}] skipped: {self.reason}"


class LayoutCheckError(AssertionError):
    """Raised by :meth:`LayoutCheckReport.raise_for_findings`; its message lists every finding."""


@dataclass(frozen=True)
class LayoutCheckReport:
    """Every finding of one check, and the rules that did not run."""

    findings: tuple[LayoutFinding, ...]
    skipped: tuple[SkippedRule, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether no rule found a defect. A skipped rule never turns a report into a failure on its own."""
        return not self.findings

    def raise_for_findings(self) -> None:
        """Raise one :class:`LayoutCheckError` that lists every finding; do nothing when the layout is ok."""
        if not self.ok:
            raise LayoutCheckError("\n" + self.render())

    def render(self) -> str:
        """The report as plain text: a verdict line, every finding, then the skipped rules."""
        count = len(self.findings)
        verdict = "layout check: ok" if self.ok else f"layout check: {count} finding(s)"
        lines = [verdict, *(finding.render() for finding in self.findings)]
        lines.extend(rule.render() for rule in self.skipped)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()
