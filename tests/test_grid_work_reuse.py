"""Pure call-local subdivision memoization; no native or performance claims."""

from dataclasses import replace
from threading import Barrier, Lock

import pytest

from scgsim.geometry._primitives._grid_cache import (
    cached_subdivision,
    grid_work_cache_scope,
    with_grid_work_cache,
)
from scgsim.geometry._primitives.grid import (
    GridOperand,
    _SOURCE_AUDITS,
    _source_recipe_map,
    geometry_grid_scope,
    source_grid_audits,
)
from scgsim.geometry.models.grid import GeometryGridContext
from scgsim.preparation import preparation_resources


BASE = GridOperand("A", ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)))
CONTEXT = GeometryGridContext(0.1)


def test_exact_reuse_nested_scope_fresh_calls_and_exception_cleanup():
    calls = []

    def build():
        calls.append(1)
        return (("immutable-subdivision", len(calls)),)

    @with_grid_work_cache
    def nested():
        return cached_subdivision((BASE,), CONTEXT, apply_grid=True, build=build)

    first = nested()
    assert nested() != first
    with grid_work_cache_scope():
        second = nested()
        with grid_work_cache_scope():
            assert nested() is second
    assert len(calls) == 3
    with pytest.raises(RuntimeError, match="scope failure"):
        with grid_work_cache_scope():
            nested()
            raise RuntimeError("scope failure")
    nested()
    assert len(calls) == 5
    # No active scope means no reuse.
    cached_subdivision((BASE,), CONTEXT, apply_grid=True, build=build)
    cached_subdivision((BASE,), CONTEXT, apply_grid=True, build=build)
    assert len(calls) == 7


@pytest.mark.parametrize(
    "operands,context,apply_grid",
    [
        ((replace(BASE, operand_id="B"),), CONTEXT, True),
        (
            (replace(BASE, exterior=((0.0, 0.0), (3.0, 0.0), (2.0, 2.0))),),
            CONTEXT,
            True,
        ),
        (
            (replace(BASE, holes=(((0.5, 0.5), (1.0, 0.5), (0.5, 1.0)),)),),
            CONTEXT,
            True,
        ),
        ((replace(BASE, physical=False),), CONTEXT, True),
        ((replace(BASE, contact=False),), CONTEXT, True),
        ((BASE,), GeometryGridContext(0.2), True),
        ((BASE,), CONTEXT, False),
        ((replace(BASE, exterior=((-0.0, 0.0), *BASE.exterior[1:])),), CONTEXT, True),
    ],
)
def test_complete_key_distinguishes_inputs(operands, context, apply_grid):
    calls = []

    def build():
        calls.append(1)
        return (len(calls),)

    with grid_work_cache_scope():
        original = cached_subdivision((BASE,), CONTEXT, apply_grid=True, build=build)
        changed = cached_subdivision(
            operands, context, apply_grid=apply_grid, build=build
        )
        assert changed != original
        assert (
            cached_subdivision(operands, context, apply_grid=apply_grid, build=build)
            is changed
        )
    assert len(calls) == 2


def test_order_and_failure_are_not_cached():
    other = replace(BASE, operand_id="B")
    calls = []

    def build():
        calls.append(1)
        return (len(calls),)

    with grid_work_cache_scope():
        assert cached_subdivision(
            (BASE, other), CONTEXT, apply_grid=True, build=build
        ) != cached_subdivision((other, BASE), CONTEXT, apply_grid=True, build=build)
        for _ in range(2):
            with pytest.raises(ValueError, match="builder failure"):

                def failed():
                    calls.append(1)
                    raise ValueError("builder failure")

                cached_subdivision((BASE,), CONTEXT, apply_grid=True, build=failed)
        assert cached_subdivision((BASE,), CONTEXT, apply_grid=True, build=build) == (
            5,
        )


def test_concurrent_misses_and_per_call_audits_remain_independent():
    barrier = Barrier(2)
    lock = Lock()
    calls = []

    def build():
        with lock:
            calls.append(1)
        barrier.wait()
        return (("immutable",),)

    def recipe(index):
        value = cached_subdivision((BASE,), CONTEXT, apply_grid=True, build=build)
        # Audit ownership is outside the cached pure builder, once per invocation.
        _SOURCE_AUDITS.get().append({"recipe": index})
        return value

    with (
        preparation_resources(max_workers=2),
        geometry_grid_scope(CONTEXT),
        grid_work_cache_scope(),
    ):
        assert _source_recipe_map(recipe, (0, 1)) == [
            (("immutable",),),
            (("immutable",),),
        ]
        assert len(calls) == 2
        assert source_grid_audits() == ({"recipe": 0}, {"recipe": 1})
        assert _source_recipe_map(recipe, (2, 3)) == [
            (("immutable",),),
            (("immutable",),),
        ]
        assert len(calls) == 2
        assert source_grid_audits() == tuple({"recipe": i} for i in range(4))


def test_compiler_build_owns_cache_scope_without_executing_geometry():
    import ast
    from pathlib import Path

    path = Path(__file__).parents[1] / "src/scgsim/geometry/compiler/builder.py"
    module = ast.parse(path.read_text())
    owner = next(
        n
        for n in module.body
        if isinstance(n, ast.ClassDef) and n.name == "SemanticGeometryBuilder"
    )
    build = next(
        n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "build"
    )
    assert any(
        isinstance(n, ast.Name) and n.id == "with_grid_work_cache"
        for n in build.decorator_list
    )
    assert any(
        isinstance(n, ast.ImportFrom)
        and n.module == "scgsim.geometry._primitives._grid_cache"
        and any(a.name == "with_grid_work_cache" for a in n.names)
        for n in module.body
    )
