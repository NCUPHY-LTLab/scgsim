"""Levels, slabs and active sets (design brief section 2, stage S2).

Owns the Z structure of the model: the sorted distinct levels, the slab between
each consecutive pair, which prisms are active in each slab and which sheets lie
on each level.  Levels are the exact declared Z values; no Z threshold merges
or drops them, so slabs thinner than the grid are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from scgsim.geometry.sgb.exact.families import CurveFamilyRecord
from scgsim.geometry.sgb.exact.prisms import PrismRecord


@dataclass(frozen=True)
class LevelStructure:
    """The Z structure of one model."""

    levels: tuple[Fraction, ...]
    slab_active: tuple[
        tuple[int, ...], ...
    ]  # prism indices active in slab i = (levels[i], levels[i+1])
    level_sheets: tuple[tuple[int, ...], ...]  # sheet prism indices on level i
    level_families: tuple[tuple[int, ...], ...]  # family indices targeting level i

    @property
    def slab_count(self) -> int:
        """Number of slabs (one fewer than the levels)."""
        return len(self.levels) - 1


def build_levels(
    prisms: tuple[PrismRecord, ...],
    families: tuple[CurveFamilyRecord, ...] = (),
) -> LevelStructure:
    """Derive levels, active sets, sheets per level and family targets."""
    heights = {p.z0 for p in prisms} | {p.z1 for p in prisms}
    for family in families:
        heights.update(family.target_levels)
    levels = tuple(sorted(heights))
    index = {z: i for i, z in enumerate(levels)}
    slab_active = tuple(
        tuple(
            k
            for k, prism in enumerate(prisms)
            if not prism.is_sheet and prism.z0 <= lo and prism.z1 >= hi
        )
        for lo, hi in zip(levels, levels[1:], strict=False)
    )
    level_sheets: list[list[int]] = [[] for _ in levels]
    for k, prism in enumerate(prisms):
        if prism.is_sheet:
            level_sheets[index[prism.z0]].append(k)
    level_families: list[list[int]] = [[] for _ in levels]
    for k, family in enumerate(families):
        for z in family.target_levels:
            level_families[index[z]].append(k)
    return LevelStructure(
        levels,
        slab_active,
        tuple(tuple(s) for s in level_sheets),
        tuple(tuple(f) for f in level_families),
    )
