"""S6 to S8: the shared boundary representation and what each route makes of it.

In: the ``LabelledArrangements``.
Out: the ``Topology`` (vertices, edges, faces, walls and volumes with content-addressed ids) and the ``RoutedModel``
(emitted volumes and faces, decision-D5 names and the primary groups of one route).
Next: ``checks`` validates it and ``output`` writes it.

* ``assembly``  vertices, edges, horizontal faces, walls, volumes and shells
* ``routes``    Route A, B and C as labelling policies, the names and the physical groups
"""
