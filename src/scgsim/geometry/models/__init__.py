"""Typed records that cross stages, and the reads of them that more than one stage makes: the build input and its entities, the grid context, physical layers, representation, junctions and regions.

In: nothing; this folder imports no other geometry code, so every stage can build and read these records.
Out: the values one stage hands to the next, each imported from its defining module.
Next: ``source`` builds them; ``interpretation``, ``junctions`` and ``sgb`` read them.
"""
