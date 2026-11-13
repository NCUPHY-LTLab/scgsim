"""The physical interpretation of a prepared input and its vacuum enclosure.

In: the prepared ``GeometryBuildInput`` and stack, and the selected approximation profile.
Out: the effective input and stack, which carry the physical representation plan and the automatic vacuum enclosure, with the rectangular junctions derived as its last step.
Next: ``junctions`` (called at the end of the interpretation), then ``sgb`` builds from the effective input.
"""
