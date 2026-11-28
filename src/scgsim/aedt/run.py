"""The command line entry point ``python -m scgsim.aedt.run``, which runs one prepared handoff."""

from __future__ import annotations

from .runtime.transaction import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
