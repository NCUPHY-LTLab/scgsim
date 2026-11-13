"""The source stage: from a layout and a stack to a prepared snapshot.

In: a layout (components, GDS) and a stack, with the declared grid, vacuum region and port sheets.
Out: a ``GeometryPlan`` and its snapshot, whose ``GeometryBuildInput`` and stack keep every input identity and carry the hashes the later stages record.
Next: ``interpretation`` reads the prepared input.
"""
