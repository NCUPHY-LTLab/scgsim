"""Source exterior-chain selection shared by Q3D authoring and offline readers.

This module owns no native session and never chooses a nearest electrode.
"""

from typing import Any


def q3d_terminal_segments(body: Any, side: str) -> list[int]:
    """Select one contiguous source exterior chain, never a hole or nearest edge."""
    if side not in {"+X", "-X", "+Y", "-Y"}:
        raise ValueError("Q3D Sheet terminals require an in-plane +/-X or +/-Y side")
    ring = body.exterior_um
    if ring[0] == ring[-1]:
        ring = ring[:-1]
    axis = 0 if side[-1] == "X" else 1
    extreme = (max if side[0] == "+" else min)(point[axis] for point in ring)
    selected = [
        index
        for index, point in enumerate(ring)
        if point[axis] == extreme and ring[(index + 1) % len(ring)][axis] == extreme
    ]
    starts = [index for index in selected if (index - 1) % len(ring) not in selected]
    if not selected or len(starts) != 1:
        raise ValueError(
            f"Q3D Sheet terminal {body.body_id!r}/{side} has no unambiguous exterior chain"
        )
    return selected
