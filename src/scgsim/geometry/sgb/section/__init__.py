"""SGB's 2D models: the plane section of a validated 3D model along a cut line (issue 188).

A 2D model stands for an infinitely long, uniform line with one cross-section.  SGB builds it as the exact
plane section of its own validated 3D model, held in memory, so the Entities, Nets, materials, surface kinds and
Surface-EPR bands of the 2D model are the 3D model's by construction.

* ``frame``      the cut line on the model grid and the section's frame (``u`` along the cut, ``z`` up)
* ``rules``      the cut rules and their located refusals
* ``traces``     the traces of the emitted faces and walls on the cut
* ``assembly``   the section's cells and regions, its vertices, edges, faces and loops
* ``records``    the 2D model, :class:`SectionModel`, and its records
* ``validation`` the section's structural invariants
* ``build``      the entry :func:`build_section_model`, exported here
"""

from scgsim.geometry.sgb.section.build import build_section_model
from scgsim.geometry.sgb.section.records import SectionModel

__all__ = ["SectionModel", "build_section_model"]
