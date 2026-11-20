"""Surface-EPR band geometry after the specs: their SGB inset requests and the banded build.

In: the normalized ``MA``/``MS``/``SA`` specs of the plan's snapshot, read by the rules of ``source.surface_epr_specs``; for the banded build, the effective ``GeometryBuildInput``, its ``RepresentationContext`` and the route (from ``palace/mesh``, the AEDT model preparation or ``geometry.sections``, which call it).
Out: the SGB inset requests of the specs' margins, and the banded build's files (``SgbArtifacts``: XAO, BREP and sidecar), which it builds by calling ``sgb.build_sgb_geometry`` with those requests, or its validated model in memory.
Next: ``palace/mesh`` meshes the banded build, the AEDT preparation exports it, and ``geometry.sections`` cuts the model in memory.

The specs are the plan's and their rules are the source stage's (``scgsim.geometry.source.surface_epr_specs``);
this folder comes after them.  ``insets`` turns the margins of normalized specs into SGB inset requests, and
``build`` builds a route's geometry with those bands (with the unbanded first build that
``inset_margins_by_entity_um`` needs).  Which faces a backend integrates and how stay with the backend.  The folder sits beside the shared geometry stages, but it is
not a stage before SGB: Palace calls it, and it calls SGB.
"""

from scgsim.geometry.surface_epr.build import build_banded_geometry, build_banded_model
from scgsim.geometry.surface_epr.insets import inset_requests

__all__ = [
    "build_banded_geometry",
    "build_banded_model",
    "inset_requests",
]
