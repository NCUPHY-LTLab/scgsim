"""Command-line entry point: `python -m scgsim.aedt.run`.

Contains no logic of its own. The hand-off is executed by `runtime.transaction`,
which owns the AEDT session, so importing this module never starts AEDT.
"""

from __future__ import annotations

from .runtime.transaction import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
