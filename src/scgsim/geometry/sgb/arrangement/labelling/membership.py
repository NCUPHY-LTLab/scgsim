"""Area-weighted membership of dissolved cells (design brief section 14).

For every cell of a dissolved arrangement and for every source entity that
contributes to it, the fraction of the cell's area covered by that entity's
footprint.  Shares are non-exclusive (an overlap counts for each entity) and can
sum to more than 1.  They are computed from the raw cells -- each lies wholly
inside or outside every footprint -- with integer chord areas, and unioned
through dissolution, so dissolving a seam never loses a contributor.
"""

from __future__ import annotations

from scgsim.geometry.sgb.arrangement.dcel import Arrangement
from scgsim.geometry.sgb.arrangement.labelling.dissolve import Dissolved

SHARE_DIGITS = 12


def membership_shares(
    raw_arr: Arrangement, dissolved: Dissolved, raw_entities: list[tuple[str, ...]]
) -> list[dict[str, float]]:
    """Entity -> covered area fraction for every face of the dissolved arrangement.

    ``raw_entities[i]`` lists the entities whose footprint covers raw face ``i + 1``.
    """
    count = len(dissolved.arrangement.faces)
    total = [0] * count
    covered: list[dict[str, int]] = [{} for _ in range(count)]
    for raw_face in range(1, len(raw_arr.faces)):
        new_face = dissolved.face_map[raw_face]
        area = raw_arr.face_twice_area(raw_face)
        total[new_face] += area
        for entity in raw_entities[raw_face - 1]:
            covered[new_face][entity] = covered[new_face].get(entity, 0) + area
    return [
        {e: round(c / total[f], SHARE_DIGITS) for e, c in sorted(covered[f].items())}
        if total[f]
        else {}
        for f in range(count)
    ]
