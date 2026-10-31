"""Pick which edges of a Q3D sheet body a source or sink terminal sits on.

Shared by Q3D authoring and by readers of saved results, so both choose the same
edges. It owns no AEDT session, and it never guesses by picking the nearest
electrode: a side with no clear single answer is an error.
"""

from typing import Any


def q3d_terminal_segments(body: Any, side: str) -> list[int]:
    """Return indices of the body's outer-ring edges that lie on `side`.

    An edge belongs to the chain when both of its endpoints sit at the extreme
    coordinate of that side (the largest X for "+X", and so on). The edges must form
    one contiguous chain. Holes are ignored, and so is any other edge, however close.
    """
    if side not in {"+X", "-X", "+Y", "-Y"}:
        raise ValueError("Q3D Sheet terminals require an in-plane +/-X or +/-Y side")
    ring = body.exterior_um
    if ring[0] == ring[-1]:
        ring = ring[:-1]  # drop the repeated closing point so edges are not doubled
    axis = 0 if side[-1] == "X" else 1
    extreme = (max if side[0] == "+" else min)(point[axis] for point in ring)
    selected = [
        index
        for index, point in enumerate(ring)
        if point[axis] == extreme and ring[(index + 1) % len(ring)][axis] == extreme
    ]
    # A single contiguous chain has exactly one edge whose predecessor is not in it.
    starts = [index for index in selected if (index - 1) % len(ring) not in selected]
    if not selected or len(starts) != 1:
        raise ValueError(
            f"Q3D Sheet terminal {body.body_id!r}/{side} has no unambiguous exterior chain"
        )
    return selected
