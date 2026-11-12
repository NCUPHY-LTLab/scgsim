"""Tables and excerpts for the section "按型別要求 band" of ``docs/design/sgb/inset-imprint.qmd`` and ``consumer-contract.qmd``.

Run, with the Python of the checkout that contains the production SGB core (``scgsim.geometry.sgb``)::

    SGB_CORE=<sgb-core> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/inset_types.py

Three Surface-EPR types are asked for at once (``InterfaceSurfaces("MA")``, ``"MS"`` and ``"SA"``, margins 0.5 and 1.0 um)
on the two real inputs of the production tests (``tests/sgb_inputs.py``), built with ``build_sgb_geometry``:

* ``gds_pad_in_ground`` under Route B (finite conductors);
* ``thin_film_pair`` under Route A (sheets).

and on one stage-level scene, a ground plane that covers the whole die, where the faces have no gap edge.  The build is
repeated with the concrete requests that name the same faces (and the ids SGB gives them); the two models must have one
identity digest.  Outputs go to ``docs/design/sgb/figures/`` with the prefix ``inset-types-``:

* ``inset-types-table.md``    which faces each type expanded into, and which it skipped
* ``inset-types-record.md``   the ``inset_request`` record of the substrate-air request of the pad-in-ground sidecar and the
  ``band_family`` records it became
* ``inset-types-numbers.json``  the same facts as numbers, and the identity check
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inset_kit as kit  # noqa: E402  (also puts the production core's tests on sys.path)
from sgb_inputs import gds_pad_in_ground, thin_film_pair  # noqa: E402
from sgb_scenes import slab_model  # noqa: E402

from scgsim.contracts.sgb_sidecar import read_sidecar  # noqa: E402
from scgsim.geometry.sgb.build import build_sgb_geometry, emit_model  # noqa: E402
from scgsim.geometry.sgb.cutfamilies.requests import (  # noqa: E402
    ConductorSurface,
    InsetRequest,
    InterfaceSurfaces,
    SolutionInterface,
)
from scgsim.geometry.sgb.errors import InputError  # noqa: E402
from scgsim.geometry.sgb.model import build_conformal_model, build_model_from_catalogue  # noqa: E402

MARGINS = (0.5, 1.0)
TYPES = ("MA", "MS", "SA")


def type_requests() -> tuple[InsetRequest, ...]:
    """One type-level request per Surface-EPR type, id = the lower-case type."""
    return tuple(InsetRequest(t.lower(), InterfaceSurfaces(t), MARGINS) for t in TYPES)


def code(text: str) -> str:
    """Inline code for a pipe table: a ``|`` inside a cell would end the cell, so it is written as an entity."""
    return f"<code>{text.replace('|', '&#124;')}</code>"


def describe(support: dict) -> str:
    """A support record as short text: ``pad 的 top``, or ``substrate–vacuum（z = 0）``."""
    if "entity" in support:
        return f"{code(support['entity'])} 的 {code(support['side'])}"
    return f"{code(support['first_entity'])}–{code(support['second_entity'])}（z = {support['z_um']:g} µm）"


def concrete_equivalent(sidecar: dict) -> tuple[InsetRequest, ...]:
    """The concrete requests for the faces the type-level requests banded (their ``families``), with the ids SGB gave
    the families."""
    families = {
        family_id
        for request in sidecar["inset_request"].values()
        for family_id in request["families"]
    }
    requests = []
    for family_id in sorted(families):
        face = sidecar["band_family"][family_id]
        support = face["support"]
        concrete = (
            ConductorSurface(support["entity"], support["side"])
            if "entity" in support
            else SolutionInterface(
                support["first_entity"], support["second_entity"], support["z_um"]
            )
        )
        requests.append(InsetRequest(family_id, concrete, tuple(face["margins_um"])))
    return tuple(requests)


def build_both(effective, route: str, scratch: Path) -> tuple[dict, bool]:
    """The sidecar of the type-level build, and whether the concrete-request build has the same identity digest."""
    artifacts = build_sgb_geometry(
        effective.build_input,
        effective.representation,
        route=route,
        insets=type_requests(),
        output_dir=scratch / route,
    )
    sidecar = read_sidecar(artifacts.sidecar)
    typed = build_conformal_model(
        effective.build_input, effective.representation, route, insets=type_requests()
    )
    written = build_conformal_model(
        effective.build_input,
        effective.representation,
        route,
        insets=concrete_equivalent(sidecar),
    )
    return sidecar, typed.identity_digest() == written.identity_digest()


def rows_of(label: str, sidecar: dict) -> list[str]:
    """Table rows of one build: a row per request with its banded and skipped faces."""
    rows = []
    for record in sidecar["inset_request"].values():
        banded = "<br>".join(code(family_id) for family_id in record["families"])
        skipped = "<br>".join(describe(item["support"]) for item in record["skipped"])
        asked = f'InterfaceSurfaces("{record["support"]["interface_type"]}")'
        rows.append(
            f"| {label} | {code(asked)} | {banded or '無'} | {skipped or '無'} |"
        )
        label = ""
    return rows


def requests_and_families(sidecar: dict) -> dict:
    """The inset requests of a sidecar and the band families they became."""
    return {
        "inset_request": sidecar["inset_request"],
        "band_family": sidecar["band_family"],
    }


def covered_die_sidecar() -> dict:
    """A ground plane over the whole die: the concrete request is refused, the type-level request skips the face."""
    catalogue = slab_model("B", [("ground", "N1", (-40.0, -40.0, 40.0, 40.0))])
    concrete = InsetRequest("ground_top", ConductorSurface("ground", "top"), MARGINS)
    try:
        build_model_from_catalogue(catalogue, "B", insets=(concrete,))
    except InputError:
        pass
    else:
        raise SystemExit("a support without a gap edge was accepted")
    model = build_model_from_catalogue(
        catalogue, "B", insets=(InsetRequest("ma", InterfaceSurfaces("MA"), MARGINS),)
    )
    with tempfile.TemporaryDirectory(prefix="inset-types-covered-") as scratch:
        return read_sidecar(emit_model(model, Path(scratch)).sidecar)


def main() -> None:
    """Write the tables, the excerpt and the numbers."""
    numbers: dict = {"margins_um": list(MARGINS)}
    rows = [
        "| 輸入 | 請求 | 展開成的 face（family id） | 略過的 face |",
        "|---|---|---|---|",
    ]
    with tempfile.TemporaryDirectory(prefix="inset-types-") as scratch_name:
        scratch = Path(scratch_name)
        (scratch / "pad").mkdir()
        sidecar_b, same_b = build_both(gds_pad_in_ground(scratch / "pad"), "B", scratch)
        sidecar_a, same_a = build_both(thin_film_pair(), "A", scratch)
    sidecar_c = covered_die_sidecar()
    rows += rows_of("pad-in-ground，Route B", sidecar_b)
    rows += rows_of("thin-film pair，Route A", sidecar_a)
    rows += rows_of("整個 die 被 ground 蓋住，Route B", sidecar_c)
    numbers["identical_to_concrete_requests"] = {"B": same_b, "A": same_a}
    if not (same_a and same_b):
        raise SystemExit("a type-level build differs from its concrete equivalent")
    numbers["requests"] = {
        "pad_in_ground_route_b": requests_and_families(sidecar_b),
        "thin_film_pair_route_a": requests_and_families(sidecar_a),
        "covered_die_route_b": requests_and_families(sidecar_c),
    }
    sa = sidecar_b["inset_request"]["sa"]
    record = {
        "inset_request": {"sa": sa},
        "band_family": {f: sidecar_b["band_family"][f] for f in sa["families"]},
    }
    excerpt = (
        '```{.json filename="semantic_geometry_route_b.sidecar.json（節錄）"}\n'
        + json.dumps(record, indent=1, ensure_ascii=False)
        + "\n```\n"
    )
    kit.FIGURES.mkdir(parents=True, exist_ok=True)
    (kit.FIGURES / "inset-types-table.md").write_text("\n".join(rows) + "\n")
    (kit.FIGURES / "inset-types-record.md").write_text(excerpt)
    (kit.FIGURES / "inset-types-numbers.json").write_text(
        json.dumps(numbers, indent=1, ensure_ascii=False)
    )
    print("\n".join(rows))


if __name__ == "__main__":
    main()
