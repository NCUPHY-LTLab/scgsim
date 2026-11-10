"""Backend-neutral Surface-EPR geometry: normalized specs, their SGB band requests and the banded build.

``specs`` normalizes the ``MA``/``MS``/``SA`` specs and says which margins a face takes, ``insets`` turns
their margins into SGB inset requests, and ``build`` builds a route's geometry with those bands (with the
unbanded first build that ``inset_margins_by_entity_um`` needs).  Palace and AEDT import these from here;
which faces a backend integrates and how stay with the backend.
"""

from scgsim.geometry.surface_epr.build import build_banded_geometry
from scgsim.geometry.surface_epr.insets import inset_requests
from scgsim.geometry.surface_epr.specs import margins_of, normalize_surface_epr_specs

__all__ = [
    "build_banded_geometry",
    "inset_requests",
    "margins_of",
    "normalize_surface_epr_specs",
]
