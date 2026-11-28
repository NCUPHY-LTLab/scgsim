"""The faces a Surface-EPR band can lie on: the vocabulary the spec rules and SGB's band requests share.

A band lies on a conductor's sheet, on the top or bottom of a finite conductor, or on a solution interface; never
on a sidewall.  The plan checks a spec's ``face_kinds`` against it before any build, and SGB reads it to select
the faces it bands.
"""

from __future__ import annotations

from typing import Literal

ConductorSide = Literal["sheet", "top", "bottom"]
CONDUCTOR_SIDES: tuple[ConductorSide, ...] = ("sheet", "top", "bottom")
"""The faces of a conductor a band can lie on: its sheet, or the top or bottom of its finite body."""
BandedFaceKind = Literal["sheet", "top", "bottom", "interface"]
BANDED_FACE_KINDS: tuple[BandedFaceKind, ...] = ("sheet", "top", "bottom", "interface")
"""The face kinds a band can lie on: a conductor sheet, the top or bottom of a finite conductor, a solution interface."""
