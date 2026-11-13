"""Semantic geometry builder (SGB) core: the prism-based, Boolean-free pipeline.

The package derives a complete conformal solid model from prisms (a planar
footprint extruded over a Z interval) before any CAD kernel is involved.  Its inputs
are the stable outputs of the source and physical-interpretation stages: the effective
``GeometryBuildInput``, the ``RepresentationContext`` and the declared grid.

The Palace mesh stage and ``geometry.surface_epr`` (which builds the geometry with the
Surface-EPR bands its specs ask for) call it through :func:`build_sgb_geometry`.  The
junction partition of ``geometry.junctions`` reads its boundary binder,
``prisms.bound_rings``.  SGB imports nothing from ``geometry.planning``,
``geometry.interpretation`` or ``geometry.junctions``; the partition's record reaches it
through the input.

The folders are the stages of the build, in the order the work flows; each folder's
docstring says what comes in, what goes out and which stage is next.  ``exact`` is
the vocabulary all of them share:

* ``exact``        exact arithmetic and the records every stage shares: the grid, plane
                   predicates, circle math, carriers, labels, prisms, cut families and ids
* ``prisms``       S2: the effective input as prisms, levels and slabs; boundary binding
* ``bands``        S3: typed requests (inset bands of chosen faces or of every face of a Surface-EPR type)
                   and the cut families they generate; the inset support face is found in a physical
                   pre-pass
* ``arrangement``  S4 and S5: noding on the grid, the planar arrangement, cell labels and dissolution; the
                   slab and level arrangements as one labelled, content-addressed result,
                   :func:`~scgsim.geometry.sgb.arrangement.derivation.labelled.derive_labelled_arrangements`
* ``topology``     S6 to S8: vertices, edges, horizontal faces, walls (coalescing, curved walls split into
                   exact cylinders), volumes and shells; Route A, B and C as labelling policies, decision-D5
                   names and the primary physical groups
* ``checks``       S9: the structural validation (fail closed, never disabled), the sliver report (local
                   feature size on exact carriers; report only), and the load-side and OCCT checks on
                   emitted files
* ``output``       S10: numbering of the shapes in ``TopExp::MapShapes`` order and the direct,
                   byte-deterministic BREP/XAO writer (no CAD kernel, no Boolean); the versioned JSON
                   sidecar ``scgsim.sgb.sidecar.v4`` and its JSON Schema; the AEDT STEP export (OCP)
* ``section``      the 2D section model of a validated model, cut along a line

Beside them: ``model`` runs the stages and returns the validated model
:func:`~scgsim.geometry.sgb.model.build_conformal_model`, ``build`` is the public entry
:func:`build_sgb_geometry`, exported here, and ``errors`` holds the refusals.

Importing this package needs neither Gmsh nor OCP: ``checks.verification.loaded`` imports Gmsh and
``checks.verification.occt`` / ``output.exchange.step`` import OCP only when they are themselves imported or called.
"""

from scgsim.geometry.sgb.build import SgbArtifacts, build_sgb_geometry

__all__ = ["SgbArtifacts", "build_sgb_geometry"]
