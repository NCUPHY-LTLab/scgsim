"""Checks of the written geometry: the Gmsh load check and what the OCCT checks add.

* ``loaded``  -- :func:`~scgsim.geometry.sgb.output.verification.loaded.verify_loaded`, the load check the tests
  run after opening the XAO in Gmsh (the mesher trusts the files of one build and does not run it): counts, vertex positions, boundaries, face areas, solid
  volumes and physical groups against the sidecar, and re-applying the embeddings the sidecar records.
  It needs Gmsh only (the ``palace`` extra).
* ``occt``    -- the official OpenCASCADE checks (``BRepCheck_Analyzer``, ``BOPAlgo_ArgumentAnalyzer``,
  ``ShapeAnalysis``) of a written file, through OCP.  OCP is the optional ``occt`` dependency group: tests
  use it; end-user installs do not pull it.  ``sidecar_mismatches`` there is
  the OCP twin of the load-side comparison, used by the STEP export to refuse a file that does not
  belong to its sidecar.
* ``tolerances`` -- the one set of tolerances both comparisons use (vertex position, relative area and volume).
"""
