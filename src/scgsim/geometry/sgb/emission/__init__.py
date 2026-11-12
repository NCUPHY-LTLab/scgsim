"""Stage S10: the direct BREP / XAO writer (design brief decision D8).

The planned topology is serialised straight to OpenCASCADE's ``CASCADE Topology V1`` text and the XAO
wrapper -- no Gmsh/OCC API call, no Boolean, no healing.  The modules follow the data:

* ``walk``      numbers every shape in ``TopExp::MapShapes`` pre-order, which is the native tag order
* ``surfaces``  the plane or cylinder each face lies on
* ``plan``      the numbered arrays (:class:`~scgsim.geometry.sgb.emission.plan.BrepPlan`) and their checks
* ``records``   the text of every section, pure over the plan
* ``writer``    the file layout and the chunked rendering; bytes never depend on the chunk size
* ``numbers``   the platform-independent number policy; ``measures`` the planned areas and volumes
"""
