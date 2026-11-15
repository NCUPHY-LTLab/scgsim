"""The rectangular partition around a lumped port: its cut rule, exact emission and checks; the junction step of ``interpretation``.

In: the effective input from ``interpretation`` and its rectangular-junction requests, each naming a source polygon
and its two terminal nets; the bound rings of the terminals from the exact kernel (``exact``), the binding SGB uses.
Out: the input with each requested polygon cut into a central rectangle and the end pieces that belong to its
terminals, put back on the input's grid, and a ``junction_partitions`` record of the cut.
Next: ``interpretation`` returns that input, and ``sgb`` builds from it and reads the record; SGB never imports
this folder, and this folder reads nothing of SGB.
"""
