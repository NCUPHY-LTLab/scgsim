"""A caller imports every build-time refusal class from the one module the supported-scope page names."""

from __future__ import annotations

import scgsim.geometry.sgb.errors as documented
from scgsim.geometry.exact import errors as kernel

NAMES = (
    "SgbError",
    "InputError",
    "NoGapEdgeError",
    "UnsupportedCurveError",
    "RouteError",
    "OverlapError",
    "ContactError",
    "NonManifoldError",
    "ValidationError",
)
KERNEL_NAMES = ("SgbError", "InputError", "UnsupportedCurveError")


def test_every_refusal_class_is_importable_from_the_documented_module():
    """The module the supported-scope page names offers all nine classes, each a refusal of the build."""
    assert sorted(documented.__all__) == sorted(NAMES)
    for name in NAMES:
        assert issubclass(getattr(documented, name), documented.SgbError), name


def test_the_kernel_classes_listed_there_are_the_classes_the_kernel_raises():
    """A handler written against the documented module catches what the exact kernel raises."""
    for name in KERNEL_NAMES:
        assert getattr(documented, name) is getattr(kernel, name), name
