"""Geometry previews of a Palace run folder or an AEDT project, drawn from the solver's own records.

In: a Palace run folder holding ``palace.msh``, ``config.json``, the mesh manifest and the index map, or an AEDT
project.  Out: a ``GeometryPreview`` with four modes (materials, boundaries, Surface EPR, mesh), each rendered
as a contact-sheet image or as an interactive scene.  Next: a notebook displays it.
"""

from .loaders.aedt import inspect_aedt_geometry
from .loaders.palace import inspect_palace_geometry
from .preview import GeometryPreview, GeometryPreviewArtifact

__all__ = [
    "GeometryPreview",
    "GeometryPreviewArtifact",
    "inspect_aedt_geometry",
    "inspect_palace_geometry",
]
