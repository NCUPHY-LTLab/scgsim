"""The physical interpretation of a prepared input and its vacuum enclosure.

In: the prepared ``GeometryBuildInput`` and stack, and the selected approximation profile.
Out: the effective input and stack, which carry the physical representation plan and the automatic vacuum enclosure;
the input's rectangular junctions are derived on it and its sheet interfaces bound to the profile.
Next: its junction step runs ``junctions``; then ``sgb`` builds from the effective input.
"""
