"""Semantic geometry builder (SGB) core: the prism-based, Boolean-free pipeline.

The package derives a complete conformal solid model from prisms (a planar
footprint extruded over a Z interval) before any CAD kernel is involved.  It sits
beside the older ``compiler``, ``construction`` and ``planning`` packages; nothing
in those packages imports it and it imports nothing from them.  Its inputs are the
stable outputs of the source and physical-interpretation stages: the effective
``GeometryBuildInput``, the ``RepresentationContext`` and the declared grid.

Sub-packages follow the stages of the design brief, lowest layer first:

* ``records``      immutable vocabulary shared by every stage (grid, carriers,
                   prisms, labels, cut families)
* ``prisms``       S2: effective input to a prism catalogue, levels, slabs and
                   active sets
* ``arrangement.noding``       S4: iterated snap rounding on the integer grid
* ``arrangement``  S4: half-edge structure, faces with holes, arc realisation
* ``arrangement.labelling``    S5: cell labels, dissolution, membership, slab/level links
* ``arrangement.derivation``   the slab and level arrangements as one labelled, content-addressed
                   result, the contact and non-manifold checks, and the entry point
                   :func:`~scgsim.geometry.sgb.arrangement.derivation.labelled.derive_labelled_arrangements`

* ``bands``  S3: typed requests (inset bands of chosen faces or of every face of a Surface-EPR type,
                   refinement curves) and their generators; the inset support face is found in a
                   physical pre-pass
* ``topology.assembly``     S6: vertices, edges, horizontal faces, walls (coalescing, curved walls
                   split into exact cylinders), vertical edges, volumes and shells
* ``topology.routes``       S7/S8: Route A, B and C as labelling policies, decision-D5 names,
                   the primary physical groups and their unions
* ``validation``   S9: structural invariants, fail closed, never disabled
* ``checks.slivers``      S9: the sliver report (local feature size on exact carriers; report only)
* ``model``        the validated model :func:`~scgsim.geometry.sgb.model.build_conformal_model`
* ``output.emission``     S10: numbering of the shapes in ``TopExp::MapShapes`` order and the direct, byte-deterministic
                   BREP/XAO writer (no CAD kernel, no Boolean)
* ``sidecar``      the versioned JSON sidecar ``scgsim.sgb.sidecar.v3`` and its JSON Schema
* ``checks.verification`` the load-side check the mesher runs on the opened XAO (Gmsh) and the official OCCT checks (OCP)
* ``output.exchange``     the AEDT STEP export of the emitted solids (OCP)
* ``build``        the public entry :func:`build_sgb_geometry`, exported here

with ``identity`` (content-addressed ids), ``plane`` (exact integer geometry),
``stopwatch`` (wall-time readings) and ``errors`` beside them.

Importing this package needs neither Gmsh nor OCP: ``checks.verification.loaded`` imports Gmsh and
``checks.verification.occt`` / ``output.exchange.step`` import OCP only when they are themselves imported or called.
"""

from scgsim.geometry.sgb.build import SgbArtifacts, build_sgb_geometry

__all__ = ["SgbArtifacts", "build_sgb_geometry"]
