"""SGB's 2D models: the plane section of a validated 3D model along a cut line, and the line it shows.

A 2D model stands for an infinitely long, uniform line with one cross-section.  SGB builds it as the exact
plane section of its own validated 3D model, held in memory, so the Entities, Nets, materials, surface kinds and
Surface-EPR bands of the 2D model are the 3D model's by construction.

In: the validated ``ConformalModel`` (from ``model``) and a ``CutLine``.
Out: the ``SectionModel``: the section's vertices, edges, faces and conductor regions with the 3D groups, Surface-EPR rows and band families the cut crosses, validated in 2D; and its classification, the ``LineCrossSection``; nothing is written.
Next: ``geometry.sections`` returns the ``LineCrossSection``; no stage of the package reads the section yet.

* ``frame``      the cut line on the model grid and the section's frame (``u`` along the cut, ``z`` up)
* ``rules``      the cut rules and their located refusals
* ``traces``     the traces of the emitted faces and walls on the cut
* ``assembly``   the section's cells and regions, its vertices, edges, faces and loops, and its conductor regions
* ``records``    the 2D model, :class:`SectionModel`, and its records
* ``validation`` the section's structural invariants
* ``build``      the entry :func:`build_section_model`, exported here
* ``lines``      the classification of a section into a ``LineCrossSection``, :func:`line_cross_section`,
  exported here
"""

from scgsim.geometry.sgb.section.build import build_section_model
from scgsim.geometry.sgb.section.lines import line_cross_section
from scgsim.geometry.sgb.section.records import SectionModel

__all__ = ["SectionModel", "build_section_model", "line_cross_section"]
