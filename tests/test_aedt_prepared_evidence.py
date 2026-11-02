"""File-only prepared CLC evidence contracts; no native execution is represented."""

from __future__ import annotations

import copy
import hashlib

import pytest

from scgsim.aedt.epr.expression_codec import (
    clc_block,
    compile_named_expression,
    parse_native_expression_readback,
    validated_expression_readback,
)
from scgsim.aedt.epr.fields import field_integral_operations


def _compile(purpose, operations):
    return compile_named_expression(
        purpose=purpose,
        operations=operations,
        solution="Setup1 : LastAdaptive",
        phase_degrees=0.0,
        dependencies={},
        selection={},
    )


def _artifact(root, blocks, filename="shared.clc"):
    # These are schema fixtures made from producer CLC, not native observations.
    payload = "".join(block for _, block in blocks).encode()
    root.mkdir(parents=True, exist_ok=True)
    (root / filename).write_bytes(payload)
    return {
        "status": "native_clc_saved_and_field_selection_read_back",
        "api": "FieldsReporter.SaveNamedExpressions",
        "path": "metadata/epr_expressions/" + filename,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "definitions": parse_native_expression_readback(payload),
    }


def _expression(compiled, artifact):
    expression, blocks, _ = compiled
    names = [name for name, _ in blocks]
    return {
        **expression,
        "definition_status": "native_readback_verified",
        "native_readback": {
            **artifact,
            "expression_definition_names": names,
            "definitions": {name: artifact["definitions"][name] for name in names},
        },
    }


def _aggregate(artifact):
    return {
        "native_expression_readback": {
            **artifact,
            "definition_names": list(artifact["definitions"]),
        }
    }


def _surface():
    return compile_named_expression(
        purpose="surface",
        operations=field_integral_operations(
            quantity="electric_normal",
            selection_name="Carrier",
            adjacent_side=True,
            normal_vector=(0, 0, 1),
        ),
        solution="Setup1 : LastAdaptive",
        phase_degrees=0.0,
        dependencies={},
        selection={"selection_name": "Carrier"},
        adjacent_selection_name="Carrier",
    )


def test_shared_clc_allows_unrelated_definitions(tmp_path):
    """Allow unrelated definitions beside the required field expressions."""
    selected = _surface()
    extra = _compile("unrelated", ["NameOfExpression('UnrelatedMissingHelper')"])
    artifact = _artifact(tmp_path, selected[1] + extra[1])
    expression = _expression(selected, artifact)
    for cache in ({}, _aggregate(artifact)):
        result = validated_expression_readback(
            [expression], cache, evidence_root=tmp_path
        )
        assert set(result) == {name for name, _ in selected[1]}


def test_named_helper_closure_is_transitive(tmp_path):
    """Retain every transitively required named expression helper."""
    leaf = _compile("leaf", ["Scalar_Constant(2)"])
    helper = _compile("helper", [f"NameOfExpression('{leaf[0]['name']}')"])
    root = _compile("root", [f"NameOfExpression('{helper[0]['name']}')"])
    artifact = _artifact(tmp_path, root[1] + helper[1] + leaf[1])
    result = validated_expression_readback(
        [_expression(root, artifact)], {}, evidence_root=tmp_path
    )
    assert set(result) == {c[0]["name"] for c in (root, helper, leaf)}


def test_missing_helper_rejected_even_with_matching_root_record(tmp_path):
    """Reject a missing transitive helper despite an unchanged root expression."""
    root = _compile("root", ["NameOfExpression('MissingHelper')"])
    artifact = _artifact(tmp_path, root[1])
    with pytest.raises(RuntimeError, match="unresolved"):
        validated_expression_readback(
            [_expression(root, artifact)], {}, evidence_root=tmp_path
        )


def test_builtin_magnetic_quantity_is_not_a_missing_named_helper(tmp_path):
    """Accept the built-in magnetic quantity without requiring a named definition."""
    root = _compile(
        "magnetic",
        field_integral_operations(
            quantity="magnetic_energy",
            selection_name="Region",
        ),
    )
    artifact = _artifact(tmp_path, root[1])
    assert set(
        validated_expression_readback(
            [_expression(root, artifact)],
            {},
            evidence_root=tmp_path,
        )
    ) == {root[0]["name"]}


def test_same_name_conflict_in_shared_file_rejected(tmp_path):
    """Reject different definitions that reuse one name within a shared artifact."""
    root = _surface()
    name = root[0]["name"]
    payload = "".join(block for _, block in root[1]) + clc_block(
        name, ["Scalar_Constant(9)"]
    )
    with pytest.raises(RuntimeError, match="conflicts"):
        parse_native_expression_readback(payload.encode())


def test_conflicting_definition_across_artifacts_rejected(tmp_path):
    """Reject conflicting named definitions across expression artifacts."""
    first = _compile("one", ["Scalar_Constant(1)"])
    second = _compile("two", ["Scalar_Constant(2)"])
    a = _artifact(
        tmp_path,
        first[1] + [("Shared", clc_block("Shared", ["Scalar_Constant(3)"]))],
        "one.clc",
    )
    b = _artifact(
        tmp_path,
        second[1] + [("Shared", clc_block("Shared", ["Scalar_Constant(4)"]))],
        "two.clc",
    )
    with pytest.raises(RuntimeError, match="conflict"):
        validated_expression_readback(
            [_expression(first, a), _expression(second, b)], {}, evidence_root=tmp_path
        )


@pytest.mark.parametrize(
    "damage",
    [
        "sha256",
        "bytes",
        "status",
        "api",
        "path",
        "definition",
        "parent",
        "definition_status",
    ],
)
def test_required_evidence_integrity_is_preserved(tmp_path, damage):
    """Reject each tested corruption of required prepared evidence."""
    root = _surface()
    artifact = _artifact(tmp_path, root[1])
    expression = _expression(root, artifact)
    record = expression["native_readback"]
    if damage == "definition":
        record["definitions"][root[0]["name"]] = {"sha256": "changed"}
    elif damage == "parent":
        expression["name"] = "AbsentParent"
    elif damage == "definition_status":
        expression["definition_status"] = "authored_and_loaded"
    else:
        record[damage] = {
            "sha256": "0" * 64,
            "bytes": record["bytes"] + 1,
            "status": "unavailable",
            "api": "OtherAPI",
            "path": "metadata/epr_expressions/../escape.clc",
        }[damage]
    with pytest.raises(RuntimeError):
        validated_expression_readback([expression], {}, evidence_root=tmp_path)


def test_changed_helper_content_rejected_by_recorded_definition(tmp_path):
    """Reject helper content that differs from its recorded definition."""
    root = _surface()
    artifact = _artifact(tmp_path, root[1])
    expression = _expression(root, artifact)
    helper = root[1][0][0]
    expression["native_readback"]["definitions"][helper] = {"sha256": "changed"}
    with pytest.raises(RuntimeError, match="differs"):
        validated_expression_readback([expression], {}, evidence_root=tmp_path)


def test_aggregate_cannot_change_required_definition(tmp_path):
    """Reject an aggregate artifact that changes a required named definition."""
    root = _surface()
    artifact = _artifact(tmp_path, root[1])
    cache = copy.deepcopy(_aggregate(artifact))
    cache["native_expression_readback"]["definitions"][root[0]["name"]] = {
        "sha256": "changed"
    }
    with pytest.raises(RuntimeError):
        validated_expression_readback(
            [_expression(root, artifact)], cache, evidence_root=tmp_path
        )


def test_symlink_escape_rejected(tmp_path):
    """Reject required evidence that escapes the prepared workspace through a symlink."""
    root = _surface()
    evidence_root = tmp_path / "evidence"
    artifact = _artifact(tmp_path / "outside", root[1])
    evidence_root.mkdir()
    (evidence_root / "shared.clc").symlink_to(tmp_path / "outside/shared.clc")
    with pytest.raises(RuntimeError, match="contained"):
        validated_expression_readback(
            [_expression(root, artifact)], {}, evidence_root=evidence_root
        )


def test_unsupported_required_reference_syntax_is_rejected(tmp_path):
    """Reject unsupported syntax in a required named-expression reference."""
    root = _compile("root", ["NameOfExpression(MissingHelper)"])
    artifact = _artifact(tmp_path, root[1])
    with pytest.raises(RuntimeError, match="grammar"):
        validated_expression_readback(
            [_expression(root, artifact)], {}, evidence_root=tmp_path
        )


def test_identical_repeated_definition_is_allowed(tmp_path):
    """Accept identical repeated definitions of the same required expression."""
    root = _surface()
    artifact = _artifact(tmp_path, root[1] + root[1])
    assert set(
        validated_expression_readback(
            [_expression(root, artifact)],
            {},
            evidence_root=tmp_path,
        )
    ) == {name for name, _ in root[1]}


def test_required_helper_cycle_rejected(tmp_path):
    """Reject a cycle among required named-expression helpers."""
    root = _compile("root", ["NameOfExpression('Helper')"])
    helper = ("Helper", clc_block("Helper", [f"NameOfExpression('{root[0]['name']}')"]))
    artifact = _artifact(tmp_path, root[1] + [helper])
    with pytest.raises(RuntimeError, match="cycle"):
        validated_expression_readback(
            [_expression(root, artifact)], {}, evidence_root=tmp_path
        )
