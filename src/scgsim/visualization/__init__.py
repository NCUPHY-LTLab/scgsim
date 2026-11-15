"""Geometry previews of a Palace run folder or an AEDT project, drawn from the solver's own records.

In: a Palace run folder holding ``palace.msh``, ``config.json``, the mesh manifest and the index map, or an AEDT
project.  Out: a ``GeometryPreview`` with four modes (materials, boundaries, Surface EPR, mesh).  A mode renders
as a contact-sheet image or as an interactive scene, or reports why it is unavailable: an AEDT project has no
Surface-EPR and no mesh mode.  Next: a notebook displays it.

``loaders`` reads a source into the records of ``scene``; ``preview`` holds them as the ``GeometryPreview`` a
user gets; ``rendering`` draws a mode as images or as an interactive scene.
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
