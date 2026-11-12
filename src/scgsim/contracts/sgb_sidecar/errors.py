"""The error of the shared sidecar reader."""


class SidecarError(ValueError):
    """The document is not a sidecar this definition describes: not JSON, another version, or a structure or
    reference that does not hold.  The message says what failed and where."""
