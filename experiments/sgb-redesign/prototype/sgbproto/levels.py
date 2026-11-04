"""Levels, slabs and active sets (brief section 2, stage S2).

Owns the Z structure of the model: the sorted distinct levels, the slab
between each consecutive pair, which prisms are active in each slab and which
sheets lie on each level. Pure function of the prism and family records.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from sgbproto.records import CurveFamilyRecord, PrismRecord


@dataclass(frozen=True)
class LevelStructure:
    levels: tuple[Fraction, ...]
    slab_active: tuple[tuple[int, ...], ...]  # prism indices active in slab i = (levels[i], levels[i+1])
    level_sheets: tuple[tuple[int, ...], ...]  # sheet prism indices on level i
    level_families: tuple[tuple[int, ...], ...]  # family indices targeting level i

    @property
    def slab_count(self) -> int:
        return len(self.levels) - 1


def build_levels(prisms: tuple[PrismRecord, ...], families: tuple[CurveFamilyRecord, ...] = ()) -> LevelStructure:
    """Derive levels, active sets, sheets-per-level and family targets."""
    zs = {p.z0 for p in prisms} | {p.z1 for p in prisms}
    for fam in families:
        zs.update(fam.target_levels)
    levels = tuple(sorted(zs))
    index = {z: i for i, z in enumerate(levels)}
    slab_active = []
    for i in range(len(levels) - 1):
        lo, hi = levels[i], levels[i + 1]
        slab_active.append(
            tuple(k for k, p in enumerate(prisms) if not p.is_sheet and p.z0 <= lo and p.z1 >= hi)
        )
    level_sheets: list[list[int]] = [[] for _ in levels]
    for k, p in enumerate(prisms):
        if p.is_sheet:
            level_sheets[index[p.z0]].append(k)
    level_families: list[list[int]] = [[] for _ in levels]
    for k, fam in enumerate(families):
        for z in fam.target_levels:
            level_families[index[z]].append(k)
    return LevelStructure(
        levels,
        tuple(slab_active),
        tuple(tuple(s) for s in level_sheets),
        tuple(tuple(f) for f in level_families),
    )
