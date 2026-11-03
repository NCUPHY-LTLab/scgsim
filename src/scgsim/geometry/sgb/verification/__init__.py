"""Checks of the written geometry: what the mesher runs on load and what the OCCT checks add.

* ``loaded``  -- :func:`~scgsim.geometry.sgb.verification.loaded.verify_loaded`, the load-side check the
  mesher calls after opening the XAO in Gmsh: counts, vertex positions, boundaries, face areas, solid
  volumes and physical groups against the sidecar, and re-applying the embeddings the sidecar records.
  It needs Gmsh only (the ``palace`` extra).
* ``occt``    -- the official OpenCASCADE checks (``BRepCheck_Analyzer``, ``BOPAlgo_ArgumentAnalyzer``,
  ``ShapeAnalysis``) of a written file, through OCP.  OCP is the optional ``occt`` dependency group: tests
  and the optional strict mode use it; end-user installs do not pull it.
"""
