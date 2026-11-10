"""Protect historical inventory digests and current AST import closure."""

import hashlib
import json
from pathlib import Path

import pytest

from scgsim.aedt.results.provenance import (
    RECEIPT_V3,
    runtime_source_identity,
    validate_runtime_source,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA,
    SOURCE_SCHEMA_V26,
    SOURCE_SCHEMA_V27,
    SOURCE_SCHEMA_V28,
    current_runtime_source_closure,
    current_runtime_source_paths,
    runtime_import_closure,
    runtime_source_paths,
)

# Captured from committed pre-refactor source 8b3cfcf, independently of this catalog.
HISTORICAL_INVENTORY_DIGESTS = {
    "scgsim.aedt.runtime-source.v1": "6c9f6d5e15cdc1e3b0b603338e5af5994cd25a966ccc0cea72ebd20a8e6d8491",
    "scgsim.aedt.runtime-source.v2": "5a129b0c19a8e805e1037f3bc808978ec579b77b855e95d824b10c6c8159e0dc",
    "scgsim.aedt.runtime-source.v3": "d76419a84c5f99ae0a9d7119b25942bf32bc63104b6b42cb03b14b85afe92888",
    "scgsim.aedt.runtime-source.v4": "6f49e64aca6f6730ce40838275f7086da129c748559123ea84fbc7fe39e275a5",
    "scgsim.aedt.runtime-source.v5": "7a9462dcb77034057f72b8c2e8bc64cd061d7e4fa564a74d630df0cb29721936",
    "scgsim.aedt.runtime-source.v6": "1c0bc79b99165ae2cc69d249b80bdfccd31db8029d5505304828f424df0a6f91",
    "scgsim.aedt.runtime-source.v7": "5713351ec0cc18646e4f00365887098ad9cfa5ba03a299a4670f682d082e508a",
    "scgsim.aedt.runtime-source.v8": "25b2786d3e3e9ea21cf40ed76f7af7c411bee0540a2fcd2f0b70c0f0f23f9138",
    "scgsim.aedt.runtime-source.v9": "b4ab0a0c83b7b014bf2c2293bab76d549ffb5af4c921e2aad94806b283c483cb",
    "scgsim.aedt.runtime-source.v10": "2d5c4bec8741fc431e93fa04c5fbb37b6fe30fe547a9911f844a34ab2478f659",
    "scgsim.aedt.runtime-source.v11": "298fee0381351ce4edb1d63c976f92a4a08ea22f59c9c5ccc5f4aca18fba98d3",
    "scgsim.aedt.runtime-source.v12": "298fee0381351ce4edb1d63c976f92a4a08ea22f59c9c5ccc5f4aca18fba98d3",
    "scgsim.aedt.runtime-source.v13": "ee5303242766f75de48272576cd13b5499da77ef184985cf49282eede0489fd3",
    "scgsim.aedt.runtime-source.v15": "fd6a737868f6a2f3364ed85cb6dfe2a6c643c9ee11be10a1e7f20b587a09c9eb",
    "scgsim.aedt.runtime-source.v16": "4010795f3a96585c4b08863fe86e5d1e452b43567222f7b495fab1b097e965ca",
    "scgsim.aedt.runtime-source.v17": "d6b24aa6d3a2b0963cf410695dcc1c2b097bbd5b16991c821e07d14dcd861663",
    "scgsim.aedt.runtime-source.v18": "1df95d9b3eba5823dbd1c3c878aa7c0e9163af19bf94d737a4e38b30538d3545",
    "scgsim.aedt.runtime-source.v19": "aae896865146607e39d056b218143ad0d465effda3d45e115138c9d0203990e7",
    "scgsim.aedt.runtime-source.v20": "626a931be88886a9143deb0053e4096b49611b4f70b9891142b94505677800df",
    "scgsim.aedt.runtime-source.v21": "b1092cbe72fe8a1fcb1bcbb4f31bfeafcf5f9a9eb620181c6615e2649251665e",
    "scgsim.aedt.runtime-source.v22": "02423eca9783a182a7122c5455bfd84d7a8b66571d5b3052eaf86fe57d159396",
    "scgsim.aedt.runtime-source.v23": "23fc8ab5eba22fa26389c8739b5e7d7315147f3f784e04441f45bfe71f44a1ce",
    "scgsim.aedt.runtime-source.v24": "760276beb4b6be76cb981ca94669e18405ab2de2ee0834e3fd3e5940a7e461f0",
    "scgsim.aedt.runtime-source.v25": "106ba25b2094e8de5e7b285e345c1ddd09c6d07c4de33d63559ef021aa4a8a9d",
    # Captured from the committed ac448bf V26 roster, independently of V27.
    "scgsim.aedt.runtime-source.v26": "74c0c1af7b6048b73a5ee681d5d2a247461919449a52c2ec092adf3b56049cb5",
}

V27_PATH_DIGEST = "304e9c166cc231a33de53bd345743eb6f29a5ca92f54b8b972024182b3ddc37d"
V28_PATH_DIGEST = "1ea4a853b9d06382d6dcff5659c8863457404a8373e970c6da532d16dc700bf0"
V29_PATH_DIGEST = "4f19b242dc5b726692432dc1ec3c994268faab5753bb2945b204dacb03bb7b94"
# Issue #67 decision 8 removed these V28 modules; the AEDT SGB import adds its modules and SGB's closure.
REMOVED_FROM_V28 = {
    "src/scgsim/aedt/runtime/native/hfss_lumped.py",
    "src/scgsim/aedt/specs/support.py",
    "src/scgsim/geometry/models/lumped.py",
}


@pytest.mark.parametrize(("schema", "expected"), HISTORICAL_INVENTORY_DIGESTS.items())
def test_historical_inventory_keeps_exact_ordered_paths(schema, expected):
    """Keep each historical path catalog equal to its committed baseline."""
    paths = runtime_source_paths(schema)
    encoded = json.dumps(paths, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == expected


def test_current_inventory_covers_aedt_roots_and_import_closure():
    """Bind every AEDT module, preview reader, and retained shared seed."""
    repository = Path(__file__).resolve().parents[1]
    paths = current_runtime_source_paths()
    assert SOURCE_SCHEMA == "scgsim.aedt.runtime-source.v29"
    assert len(paths) == 388
    assert paths == tuple(sorted(set(paths)))
    assert runtime_source_paths(SOURCE_SCHEMA) == paths
    assert all((repository / path).is_file() for path in paths)
    all_aedt_modules = {
        f"src/{path.relative_to(repository / 'src').as_posix()}"
        for path in (repository / "src/scgsim/aedt").rglob("*.py")
    }
    assert all_aedt_modules <= set(paths)
    v28_paths = runtime_source_paths(SOURCE_SCHEMA_V28)
    assert v28_paths is not None
    assert set(v28_paths) - set(paths) == REMOVED_FROM_V28
    assert {
        "src/scgsim/aedt/preparation/sgb_model.py",
        "src/scgsim/aedt/runtime/families/sgb_import.py",
        "src/scgsim/geometry/sgb/exchange/step.py",
    } <= set(paths) - set(v28_paths)
    assert (
        hashlib.sha256(
            json.dumps(v28_paths, separators=(",", ":")).encode()
        ).hexdigest()
        == V28_PATH_DIGEST
    )
    assert {
        "src/scgsim/visualization/loaders/aedt.py",
        "src/scgsim/visualization/loaders/aedt_receipt.py",
        "src/scgsim/geometry/__init__.py",
        "src/scgsim/semantics/route_a.py",
        "src/scgsim/semantics/snapshot.py",
    } <= set(paths)
    v26_paths = runtime_source_paths(SOURCE_SCHEMA_V26)
    assert v26_paths is not None
    assert tuple(sorted(set(v26_paths))) == v26_paths
    assert set(v26_paths) - all_aedt_modules - REMOVED_FROM_V28 <= set(paths)

    v27_paths = runtime_source_paths(SOURCE_SCHEMA_V27)
    assert v27_paths is not None
    assert len(v27_paths) == 283
    assert set(v27_paths) - REMOVED_FROM_V28 <= set(paths)
    assert (
        hashlib.sha256(
            json.dumps(v27_paths, separators=(",", ":")).encode()
        ).hexdigest()
        == V27_PATH_DIGEST
    )
    encoded = json.dumps(paths, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == V29_PATH_DIGEST

    assert current_runtime_source_closure() == paths


def test_current_runtime_identity_uses_v29_without_changing_receipt_schema():
    """Measure V29 while retaining the V3 receipt contract."""
    identity = runtime_source_identity()
    validate_runtime_source(identity, stage="actual")
    assert identity["schema_version"] == SOURCE_SCHEMA
    assert RECEIPT_V3 == "scgsim.aedt.receipt.v3"
    assert (
        tuple(item["path"] for item in identity["modules"])
        == current_runtime_source_paths()
    )


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
    for relative, source in sources.items():
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    paths = runtime_import_closure(source_root, ("src/scgsim/aedt/entry.py",))
    assert paths == tuple(f"src/{path}" for path in sorted(sources))
