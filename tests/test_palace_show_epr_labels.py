"""The detached ``show_epr`` chart keeps every Surface-EPR entry in its own category."""

from __future__ import annotations

from types import SimpleNamespace

from scgsim.palace.presentation.epr import show_epr


def surface_item(name, index, *, evaluation_kind, margin_um):
    """A Surface-EPR row as the detached result carries it; only the fields the chart reads matter."""
    return {
        "group_id": f"palace_surface_{index}",
        "index": index,
        "name": name,
        "rows": [f"{name}_row"],
        "face_kinds": ["top"],
        "nets": [],
        "lineage": {},
        "interface_kind": "MA",
        "evaluation_kind": evaluation_kind,
        "margin_um": margin_um,
        "entities": ["Al", "vacuum"],
        "participation": 1e-3,
        "inverse_q": None,
        "assumptions": {},
        "original_assumptions": {},
    }


def test_a_banded_and_an_unbanded_face_of_one_material_pair_are_two_categories():
    """Same interface type and materials, different entries: the chart must not merge them."""
    banded = surface_item(
        "MA_Al_vacuum_banded", 1, evaluation_kind="unmasked_baseline", margin_um=0.0
    )
    banded["bands"] = ["gap_band"]
    unbanded = surface_item(
        "MA_Al_vacuum_flat", 2, evaluation_kind="unmasked_baseline", margin_um=0.0
    )
    result = SimpleNamespace(
        rows=[{"mode": 1, "surface": [banded, unbanded], "bulk": [], "junction": []}]
    )
    figure = show_epr(result, mode=1)
    categories = list(figure.data[0].y)
    assert len(categories) == len(set(categories)) == 2
    assert "MA_Al_vacuum_banded" in categories[0]
    assert "MA_Al_vacuum_flat" in categories[1]
