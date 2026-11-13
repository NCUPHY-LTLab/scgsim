"""Backend-neutral Surface-EPR geometry: normalized specs, their SGB band requests and the banded build.

In: the ``MA``/``MS``/``SA`` specs of a Palace simulation (from ``palace/source``); for the banded build, the effective
``GeometryBuildInput``, its ``RepresentationContext`` and the route (from ``palace/mesh``).
Out: the normalized specs and the margins each face takes, the SGB inset requests, and the banded build's files
(``SgbArtifacts``: XAO, BREP and sidecar).
Next: ``sgb`` builds the geometry with the inset requests; ``palace/mesh`` meshes the banded build, and
``palace/source``, ``palace/config`` and ``palace/results`` read the specs and their margins.

``specs`` normalizes the ``MA``/``MS``/``SA`` specs and says which margins a face takes, ``insets`` turns
their margins into SGB inset requests, and ``build`` builds a route's geometry with those bands (with the
unbanded first build that ``inset_margins_by_entity_um`` needs).  Palace imports these from here;
which faces it integrates and how stay with Palace.
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
