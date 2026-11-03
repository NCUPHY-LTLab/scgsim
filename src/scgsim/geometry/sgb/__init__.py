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
* ``noding``       S4: iterated snap rounding on the integer grid, tile-parallel
* ``arrangement``  S4: half-edge structure, faces with holes, arc realisation
* ``labelling``    S5: cell labels, dissolution, membership, slab/level links
* ``derivation``   the slab and level arrangements as one labelled, content-addressed
                   result, the contact and non-manifold checks, and the entry point
                   :func:`~scgsim.geometry.sgb.derivation.labelled.derive_labelled_arrangements`

with ``parallel`` (per-tile runtime), ``identity`` (content-addressed ids),
``plane`` (exact integer geometry) and ``errors`` beside them.  Later sub-stages add
assembly, routes, cut-family generators and emission as further sub-packages that
consume the records defined here.
"""
