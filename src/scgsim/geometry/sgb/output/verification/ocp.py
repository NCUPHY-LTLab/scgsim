"""Whether OCP is installed, and the readable error when it is not.

Owns the one place that knows OCP (``cadquery-ocp-novtk``, the OpenCASCADE binding) is an *optional* dependency:
the ``aedt`` and ``aedt-epr`` extras install it, and so does the ``occt`` dependency group of a checkout.  A
Palace build does not need it; the OCCT checks (``occt``) and the STEP export AEDT imports
(``output.exchange.step``) do.  This module imports without OCP; the two that use it import it at module level,
so reach them only after :func:`require_ocp` has said yes.
"""

from __future__ import annotations

from importlib.util import find_spec

from scgsim.geometry.exact.errors import SgbError


class OcctUnavailable(SgbError):
    """OCP is not installed."""


def ocp_available() -> bool:
    """Whether the OCP module can be imported."""
    return find_spec("OCP") is not None


def require_ocp() -> None:
    """Raise :class:`OcctUnavailable`, naming the extra that installs it, when OCP cannot be imported."""
    if not ocp_available():
        raise OcctUnavailable(
            "the STEP export AEDT imports and the official OCCT checks need OCP (cadquery-ocp-novtk): install "
            "SCGSim with the `aedt` or `aedt-epr` extra, or, in an SCGSim checkout, `uv sync --group occt`"
        )
