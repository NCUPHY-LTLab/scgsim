"""The AEDT runtime source is a generated import closure checked by shape."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scgsim.aedt.results.provenance import (
    ENTRY_MODULES,
    SOURCE_SCHEMA,
    record_runtime_source,
    validate_runtime_source,
)
from scgsim.aedt.results.readback_dispatch import validate_result_readback
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_ROOT,
    current_runtime_source_paths,
    runtime_import_closure,
)
from scgsim.aedt.specs.common import (
    LayerImport,
    ObjectBinding,
    PdkMaterial,
    TerminalPort,
)
from scgsim.aedt.specs.hfss import FrequencySweepSpec, HfssDrivenSpec, HfssRunControl


def _write_tree(source_root: Path, sources: dict[str, str]) -> None:
    for relative, source in sources.items():
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _aedt_tree(entry_source: str) -> dict[str, str]:
    """A minimal source tree with one AEDT module and both visualization roots."""
    return {
        "scgsim/__init__.py": "",
        "scgsim/aedt/__init__.py": "",
        "scgsim/aedt/entry.py": entry_source,
        "scgsim/shared/__init__.py": "",
        "scgsim/shared/added.py": "",
        "scgsim/unused.py": "",
        "scgsim/visualization/__init__.py": "",
        "scgsim/visualization/loaders/__init__.py": "",
        "scgsim/visualization/loaders/aedt.py": "",
        "scgsim/visualization/loaders/aedt_receipt.py": "",
    }


def test_current_list_is_a_sorted_unique_closure_with_the_entry_modules():
    """The list on this checkout is sorted, unique, contained and complete."""
    paths = current_runtime_source_paths()
    assert list(paths) == sorted(set(paths))
    assert all(path.startswith("src/scgsim/") for path in paths)
    assert set(ENTRY_MODULES) <= set(paths)
    aedt_modules = {
        f"src/{path.relative_to(SOURCE_ROOT).as_posix()}"
        for path in (SOURCE_ROOT / "scgsim/aedt").rglob("*.py")
    }
    assert aedt_modules <= set(paths)


def test_the_closure_holds_no_palace_or_meshing_file():
    """AEDT provenance names AEDT's own code: a Palace-only change must not alter it.

    The import-linter independence contract covers ``scgsim.aedt`` and ``scgsim.palace`` only, so an import
    of Palace from a module both use (the visualization loaders, ``semantics``, ``presentation``, the
    contracts) passes it; this checks the closure itself.
    """
    backend = [
        path
        for path in current_runtime_source_paths()
        if path.startswith(("src/scgsim/palace/", "src/scgsim/meshing/"))
    ]
    assert backend == []


def test_adding_an_import_to_an_aedt_module_adds_its_target(tmp_path):
    """The list follows the code: no roster has to be edited."""
    source_root = tmp_path / "src"
    _write_tree(source_root, _aedt_tree(""))
    before = current_runtime_source_paths(source_root)
    assert "src/scgsim/shared/added.py" not in before
    assert "src/scgsim/unused.py" not in before

    _write_tree(source_root, {"scgsim/aedt/entry.py": "from ..shared import added\n"})
    after = current_runtime_source_paths(source_root)
    assert set(after) - set(before) == {
        "src/scgsim/shared/__init__.py",
        "src/scgsim/shared/added.py",
    }


def test_deleting_a_file_outside_the_closure_changes_nothing(tmp_path):
    """A deletion elsewhere in src no longer breaks AEDT preparation."""
    source_root = tmp_path / "src"
    _write_tree(source_root, _aedt_tree(""))
    before = current_runtime_source_paths(source_root)
    (source_root / "scgsim/unused.py").unlink()
    assert current_runtime_source_paths(source_root) == before


def test_import_closure_follows_package_relative_absolute_child_and_lazy_imports(
    tmp_path,
):
    """Resolve direct, relative, package-child, initializer, and function imports."""
    source_root = tmp_path / "src"
    sources = {
        "scgsim/__init__.py": "",
        "scgsim/aedt/__init__.py": "from . import initialized_child\n",
        "scgsim/aedt/initialized_child.py": "",
        "scgsim/aedt/entry.py": (
            "from . import relative_child\n"
            "from scgsim.shared import imported_child\n"
            "import scgsim.shared.absolute_child\n"
            "def lazy_step():\n"
            "    from ..shared.lazy_child import LazyValue\n"
        ),
        "scgsim/aedt/relative_child.py": "",
        "scgsim/shared/__init__.py": "",
        "scgsim/shared/imported_child.py": "",
        "scgsim/shared/absolute_child.py": "",
        "scgsim/shared/lazy_child.py": "",
    }
    _write_tree(source_root, sources)

    paths = runtime_import_closure(source_root, ("src/scgsim/aedt/entry.py",))
    assert paths == tuple(f"src/{path}" for path in sorted(sources))


@pytest.mark.parametrize("stage", ["prepared", "actual"])
def test_record_of_the_current_source_is_accepted(stage):
    """Both stages record the computed list under the current schema."""
    record = record_runtime_source(stage)
    validate_runtime_source(record, stage=stage)
    assert record["schema_version"] == SOURCE_SCHEMA
    assert SOURCE_SCHEMA == "scgsim.aedt.runtime-source.v30"
    assert record["stage"] == stage
    assert (
        tuple(item["path"] for item in record["modules"])
        == current_runtime_source_paths()
    )


def _resealed(record: dict[str, Any]) -> dict[str, Any]:
    """Recompute the digest so a test isolates the one property it breaks."""
    encoded = json.dumps(
        record["modules"], ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    record["content_sha256"] = hashlib.sha256(encoded).hexdigest()
    return record


def _module(path: str) -> dict[str, str]:
    module = path.removeprefix("src/").removesuffix(".py").replace("/", ".")
    return {"module": module, "path": path, "sha256": "0" * 64}


def _with_module(record: dict[str, Any], path: str) -> dict[str, Any]:
    record["modules"] = sorted(
        [*record["modules"], _module(path)], key=lambda item: item["path"]
    )
    return _resealed(record)


def _outside_src(record: dict[str, Any]) -> dict[str, Any]:
    return _with_module(record, "tests/helper.py")


def _escaping_src(record: dict[str, Any]) -> dict[str, Any]:
    return _with_module(record, "src/scgsim/../../outside.py")


def _unsorted(record: dict[str, Any]) -> dict[str, Any]:
    record["modules"].reverse()
    return _resealed(record)


def _duplicated(record: dict[str, Any]) -> dict[str, Any]:
    record["modules"].insert(0, copy.deepcopy(record["modules"][0]))
    return _resealed(record)


def _without_entry(record: dict[str, Any]) -> dict[str, Any]:
    record["modules"] = [
        item for item in record["modules"] if item["path"] != ENTRY_MODULES[1]
    ]
    return _resealed(record)


def _wrong_digest(record: dict[str, Any]) -> dict[str, Any]:
    record["modules"][0]["sha256"] = "f" * 64
    return record


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (_outside_src, "module manifest is invalid"),
        (_escaping_src, "module manifest is invalid"),
        (_unsorted, "paths are not sorted and unique"),
        (_duplicated, "paths are not sorted and unique"),
        (_without_entry, "lacks the AEDT entry modules"),
        (_wrong_digest, "content digest is invalid"),
    ],
)
def test_a_damaged_record_is_refused(damage, message):
    """Each shape property and the digest are checked on their own."""
    record = damage(record_runtime_source("actual"))
    with pytest.raises(RuntimeError, match=message):
        validate_runtime_source(record, stage="actual")


def test_a_record_of_the_other_stage_is_refused():
    """A prepared record is not accepted where execution's record belongs."""
    with pytest.raises(RuntimeError, match="actual runtime source provenance"):
        validate_runtime_source(record_runtime_source("prepared"), stage="actual")


@pytest.mark.parametrize("schema", ["v1", "v12", "v25", "v29"])
def test_an_older_schema_is_refused_with_how_to_read_it(schema):
    """Old result folders are read with the commit that wrote them."""
    record = record_runtime_source("prepared")
    record["schema_version"] = f"scgsim.aedt.runtime-source.{schema}"
    with pytest.raises(
        RuntimeError,
        match=(
            rf"runtime-source\.{schema} was written by an earlier SCGSim.*"
            "read this result with the SCGSim commit that wrote it"
        ),
    ):
        validate_runtime_source(record, stage="prepared")


@pytest.mark.parametrize("schema", ["scgsim.aedt.runtime-source.v31", "other", None])
def test_an_unknown_schema_is_refused(schema):
    """A schema that is neither current nor older is invalid, not 'earlier'."""
    record = record_runtime_source("prepared")
    record["schema_version"] = schema
    with pytest.raises(
        RuntimeError, match="prepared runtime source provenance is invalid"
    ):
        validate_runtime_source(record, stage="prepared")


def _terminal_spec() -> HfssDrivenSpec:
    return HfssDrivenSpec(
        modeling="solid",
        aedt_version="2024.2",
        mode="terminal",
        gds_path="synthetic.gds",
        project_name="Synthetic",
        design_name="Design",
        materials={
            "vacuum": PdkMaterial("vacuum", "vacuum", False, "Vacuum"),
            "silicon": PdkMaterial("silicon", "dielectric", False, "Silicon"),
            "metal": PdkMaterial("metal", "superconductor", True, None),
        },
        vacuum_material_id="vacuum",
        layer_imports=(
            LayerImport(1, 0, "SignalLayer", 0, 1),
            LayerImport(2, 0, "GroundLayer", 0, 1),
            LayerImport(3, 0, "SubstrateLayer", -10, 0),
        ),
        object_bindings=(
            ObjectBinding("Signal", 1, "signal", "metal"),
            ObjectBinding("Ground", 2, "ground", "metal"),
            ObjectBinding("Substrate", 3, "substrate", "silicon"),
        ),
        ports=(
            TerminalPort(1, "P1", "-X", ("Ground",)),
            TerminalPort(2, "P2", "+X", ("Ground",)),
        ),
        run_control=HfssRunControl(
            "Setup", "Sweep", FrequencySweepSpec(1, 2, points=3)
        ),
        region_padding_um=(0, 0, 10, 10, 10, 10),
    )


def _terminal_port_record(spec: HfssDrivenSpec, port: TerminalPort) -> dict[str, Any]:
    """Terminal evidence as AEDT reports it for the default deembed of 0 um.

    AEDT reports ``Deembed: False`` and its own renormalizing impedance. The
    checks once kept for receipts before runtime-source v19 expected
    ``Deembed: True`` and ``50ohm``, and refused this evidence.
    """
    terminal_name = f"{port.name}_T1"
    impedance = "50.0ohm"
    return {
        "index": port.index,
        "boundary": port.name,
        "terminal_excitation": terminal_name,
        "face_id": port.index,
        "face_center_um": [0, 0, 0],
        "requested": {
            "reference_objects": list(port.reference_objects),
            "renormalize": False,
            "deembed_um": port.deembed_um,
        },
        "native": {
            "excitation_names": [item.name for item in spec.ports[: port.index]],
            "terminal_names": [terminal_name],
            "reference_conductors": list(port.reference_objects),
            "reference_conductor_ids": [10 + port.index],
            "boundary_properties": {
                "Deembed": False,
                "Deembed Dist": f"{port.deembed_um:g}um",
                "Name": port.name,
                "Num Terminals": "1",
                "Renorm All Terminals": False,
                "Type": "Wave Port",
                "Wave Port Type": "Terminal",
            },
            "terminal_properties": {
                "Name": terminal_name,
                "Port Name": port.name,
                "Terminal Renormalizing Impedance": impedance,
                "Type": "Terminal",
            },
            "inactive_renormalizing_impedance": impedance,
        },
    }


def _terminal_receipt(spec: HfssDrivenSpec) -> dict[str, Any]:
    """A completed terminal receipt up to its ports; it has no setup record."""
    sweep = {
        "records": 3,
        "frequency_unit": "GHz",
        "first_frequency_ghz": 1,
        "last_frequency_ghz": 2,
        "strictly_increasing": True,
    }
    ports = [
        _terminal_port_record(spec, port)
        for port in spec.ports
        if isinstance(port, TerminalPort)
    ]
    return {
        "connected": {
            "aedt_version": spec.aedt_version,
            "pyaedt_version": spec.pyaedt_version,
        },
        "runtime_source": record_runtime_source("actual"),
        "ports": ports,
        "result_readback": {
            "terminal_st": sweep,
            "touchstone": {
                **sweep,
                "path": "terminal.s2p",
                "ports": 2,
                "port_order": [record["terminal_excitation"] for record in ports],
                "bytes": 100,
            },
        },
    }


def test_a_current_terminal_receipt_takes_the_current_terminal_checks(tmp_path):
    """Defect F2: a terminal receipt of the current schema passes the port checks.

    The receipt has no setup record, so the readback stops at the setup check
    after the terminal port checks accepted the evidence; damaged evidence
    stops at the port checks.
    """
    spec = _terminal_spec()
    receipt = _terminal_receipt(spec)
    validate_runtime_source(receipt["runtime_source"], stage="actual")
    with pytest.raises((RuntimeError, TypeError)) as refused:
        validate_result_readback(tmp_path, receipt, spec)
    assert "terminal" not in str(refused.value)

    receipt["ports"][0]["native"]["boundary_properties"]["Deembed"] = "maybe"
    with pytest.raises(RuntimeError, match="terminal native property mismatch"):
        validate_result_readback(tmp_path, receipt, spec)
