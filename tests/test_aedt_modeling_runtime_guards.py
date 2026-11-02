"""Runtime guards for AEDT modeling payload production."""

from types import SimpleNamespace

import pytest

from scgsim.aedt.specs import modeling


def test_fresh_modeling_payload_requires_a_representation_plan(monkeypatch):
    """Reject a missing fresh plan with an optimization-safe runtime guard."""
    spec = SimpleNamespace(
        modeling="solid",
        physical_layers=(),
        _historical_physical=False,
    )
    monkeypatch.setattr(modeling, "request_modeling", lambda _spec: None)

    with pytest.raises(
        RuntimeError,
        match="fresh physical modeling request did not produce a representation plan",
    ):
        modeling.modeling_payload(spec)
