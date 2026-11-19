"""Facts of a primary physical group read from the band labels of its faces, never from its name.

A group name carries its face kind and its band (``MA::pad__top::band=1``), but a consumer must not parse them back out:
the labels of the faces are the structured source (``TopoFace.bands``), and the group's membership comes from the routed model.
"""

from __future__ import annotations


def band_of_group(model, group: str) -> tuple[str, int] | None:
    """The ``(family id, band index)`` of a primary surface group, or ``None`` for an unbanded one.

    A group never mixes two families (the build stops otherwise), so its faces carry one label.
    """
    physical = model.routed.groups.get(group)
    if physical is None or physical.dim != 2:
        return None
    labels = {
        face.bands[0]
        for face in (model.topology.faces[fid] for fid in physical.ids)
        if face.bands
    }
    if not labels:
        return None
    (label,) = labels
    return label


def banded_groups(model) -> dict[str, tuple[str, int]]:
    """Every banded primary surface group of the model with its ``(family id, band index)``."""
    found = {name: band_of_group(model, name) for name in model.routed.groups}
    return {name: label for name, label in found.items() if label is not None}
