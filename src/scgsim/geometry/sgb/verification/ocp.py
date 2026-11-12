"""Whether OCP is installed, and the readable error when it is not.

Owns the one place that knows OCP (``cadquery-ocp``, the OpenCASCADE binding) is an *optional* dependency, the
``occt`` dependency group: nothing in a normal build needs it, the OCCT
checks (``occt``) and the STEP export (``exchange.step``) do.  This module imports without OCP; the two that use it
import it at module level, so reach them only after :func:`require_ocp` has said yes.
"""

from __future__ import annotations

from importlib.util import find_spec

from scgsim.geometry.sgb.errors import SgbError


class OcctUnavailable(SgbError):
    """OCP is not installed."""


def ocp_available() -> bool:
    """Whether the OCP module can be imported."""
    return find_spec("OCP") is not None


def require_ocp() -> None:
    """Raise :class:`OcctUnavailable`, naming the dependency group, when OCP cannot be imported."""
    if not ocp_available():
        raise OcctUnavailable(
            "the official OCCT checks and the STEP export need OCP (cadquery-ocp), the optional `occt` "
            "dependency group: install it with `uv sync --group occt` (or `pip install cadquery-ocp-novtk`); "
            "end-user installs do not pull it"
        )
