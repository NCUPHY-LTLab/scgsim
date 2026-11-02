"""Keep the failed Terminal port identifiable after owner-query extraction."""

import pytest

from scgsim.aedt.runtime.families import hfss_ports


def test_unavailable_saved_terminal_assignment_names_the_port():
    """Retain the original diagnostic when no native child assignment exists."""
    with pytest.raises(RuntimeError) as captured:
        hfss_ports._read_body_terminal_owners(  # noqa: SLF001 -- isolate the extracted failure boundary without native calls.
            object(), {}, {}, None, None, "readout_port"
        )

    assert str(captured.value) == (
        "saved Terminal child assignment is unavailable: 'readout_port'"
    )
