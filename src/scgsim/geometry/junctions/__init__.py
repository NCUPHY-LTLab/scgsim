"""The rectangular partition around a lumped port: its cut rule, exact emission and checks.

In: the effective input and its rectangular-junction requests, each naming a source polygon and its two terminal nets.
Out: the input with each requested polygon cut into a central rectangle and the end pieces that belong to its terminals, and a ``junction_partitions`` record of the cut.
Next: ``sgb`` builds from that input and reads the record; it never imports this folder.
"""
