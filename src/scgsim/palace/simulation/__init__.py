"""The stage a user drives: bind a prepared plan, set the problem, then mesh, write the config and hand off.

In: a ``GeometryPlanSnapshot`` and the problem's settings.  Out: a run folder with ``palace.msh``,
``config.json`` and the run script (``HandoffPlan``).  Next: the user runs Palace; ``results`` reads the
folder it returns (Electrostatic and Eigenmode; not Driven yet).  ``core`` holds what every problem shares,
``lifecycle`` which call needs which stage, ``eigenmode``, ``electrostatic`` and ``driven`` what each problem
adds, and ``drc`` the modeling check.
"""
