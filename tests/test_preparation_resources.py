"""Preparation resource ownership contracts, without geometry or native work."""

from contextvars import ContextVar
import json
from threading import Barrier, Event, get_ident

import pytest

from scgsim.diagnostics import _stage, preparation_diagnostics
from scgsim.preparation import (
    _ordered_map,
    current_preparation_workers,
    preparation_resources,
)


def test_outside_scope_is_serial_and_preserves_order():
    caller = get_ident()
    assert current_preparation_workers() is None
    assert _ordered_map(lambda n: (n, get_ident()), range(4)) == [
        (n, caller) for n in range(4)
    ]


@pytest.mark.parametrize("value", (True, False, 1.5, "3", None))
def test_non_integer_budget_rejected(value):
    with pytest.raises(TypeError, match="positive integer"):
        with preparation_resources(max_workers=value):
            pytest.fail("invalid budget entered")


@pytest.mark.parametrize("value", (0, -1))
def test_nonpositive_budget_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        with preparation_resources(max_workers=value):
            pytest.fail("invalid budget entered")


def test_parallel_output_context_and_nested_budget():
    marker = ContextVar("resource_test_marker", default="outside")
    token = marker.set("caller")
    barrier = Barrier(3)

    def task(n):
        assert marker.get() == "caller"
        marker.set("worker")
        barrier.wait(timeout=5)
        worker = get_ident()
        with preparation_resources(max_workers=30):
            assert current_preparation_workers() == 3
            nested = _ordered_map(lambda value: (value, get_ident()), range(2))
        assert nested == [(0, worker), (1, worker)]
        return n, worker

    try:
        with preparation_resources(max_workers=3):
            with preparation_resources(max_workers=1):
                assert current_preparation_workers() == 3
                result = _ordered_map(task, range(3))
            assert [row[0] for row in result] == [0, 1, 2]
            assert len({row[1] for row in result}) == 3
            assert marker.get() == "caller"
        assert current_preparation_workers() is None
    finally:
        marker.reset(token)


def test_failure_drains_owned_work_and_bounds_submission():
    barrier = Barrier(2)
    release = Event()
    finished = Event()
    consumed = []
    original = ValueError("independent task failure")

    def items():
        for n in range(100):
            consumed.append(n)
            yield n

    def task(n):
        barrier.wait(timeout=5)
        if n == 0:
            release.set()
            raise original
        assert release.wait(timeout=5)
        finished.set()
        return n

    with preparation_resources(max_workers=2):
        with pytest.raises(ValueError) as observed:
            _ordered_map(task, items())
        assert observed.value is original
        assert finished.is_set()
        assert consumed == [0, 1]
        assert _ordered_map(lambda n: n + 1, [1, 2]) == [2, 3]


@pytest.mark.parametrize(
    "threads,surface_threads", ((3, 1), (1, 3), (0, 1), (-1, 1), (1, 0), (1, -1))
)
def test_mesh_budget_rejects_before_child(threads, surface_threads, tmp_path):
    from scgsim.palace.mesh.build import _mesh_from_route_xao

    with preparation_resources(max_workers=2):
        with pytest.raises(ValueError, match="must be positive and fit"):
            _mesh_from_route_xao(
                xao_path=tmp_path / "unused.xao",
                route="B",
                records=[],
                stack={},
                mesh_path=tmp_path / "unused.msh",
                refined_mesh_size=None,
                max_mesh_size=None,
                mesh_controls={
                    "size_policy": "gmsh_default",
                    "threads": threads,
                    "surface_threads": surface_threads,
                },
            )
    assert not (tmp_path / "unused.msh").exists()


def test_one_worker_and_unscoped_native_guard_preserve_serial_behavior():
    from scgsim.preparation import _native_preparation

    with _native_preparation({"threads": 0, "surface_threads": -1}):
        assert current_preparation_workers() is None
    caller = get_ident()
    with preparation_resources(max_workers=1):
        assert _ordered_map(lambda n: get_ident(), range(3)) == [caller] * 3
        with _native_preparation({"threads": 1, "surface_threads": 1}):
            assert current_preparation_workers() == 1


def test_budget_is_required():
    with pytest.raises(TypeError, match="max_workers"):
        with preparation_resources():
            pytest.fail("implicit resource budget entered")


def test_mesh_preparation_keeps_parallel_mapping_before_native(monkeypatch, tmp_path):
    import hashlib
    from types import SimpleNamespace
    from scgsim.geometry.compiler.builder import SemanticGeometryBuilder
    from scgsim.palace.mesh.build import build_route_mesh

    class StopBeforeNative(Exception):
        pass

    barrier = Barrier(2)
    observed = []

    def task(n):
        barrier.wait(timeout=5)
        return n, get_ident()

    def construction(self, *args, **kwargs):
        observed.extend(_ordered_map(task, range(2)))
        raise StopBeforeNative

    # Stop at the real construction call boundary: no fake XAO or native result.
    monkeypatch.setattr(SemanticGeometryBuilder, "build", construction)
    source = b"file-only identity for construction wiring"
    stack = {
        "metadata": {
            "gds_sha256": hashlib.sha256(source).hexdigest(),
            "top_cell_name": "COUPON",
        }
    }
    with preparation_resources(max_workers=2):
        with pytest.raises(StopBeforeNative):
            build_route_mesh(
                component=None,
                stack=stack,
                route="B",
                output_dir=tmp_path / "run",
                source_gds_bytes=source,
                representation_context=object(),
                source_geometry_input=SimpleNamespace(
                    boundary_curves=(), boundary_reconstruction=()
                ),
            )
    assert [row[0] for row in observed] == [0, 1]
    assert len({row[1] for row in observed}) == 2


def test_threaded_diagnostics_write_complete_records(tmp_path):
    def task(n):
        with _stage("preparation.test.task"):
            return n

    with preparation_diagnostics(tmp_path, profile=False):
        with preparation_resources(max_workers=3):
            assert _ordered_map(task, range(12)) == list(range(12))
    paths = list(tmp_path.glob("*.jsonl"))
    assert len(paths) == 1
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    task_rows = [row for row in rows if row["stage"] == "preparation.test.task"]
    assert len(task_rows) == 24
    assert sum(row["event"] == "start" for row in task_rows) == 12
    assert sum(row["event"] == "end" for row in task_rows) == 12
