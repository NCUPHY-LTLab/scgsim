"""Expose source-domain junction partitioning independently of solver analysis."""

from scgsim.geometry.planning.junction_planning.contact import (
    JunctionContactRequest as JunctionContactRequest,
)
from scgsim.geometry.planning.junction_planning.contact import (
    junction_contact_inventory as junction_contact_inventory,
)
from scgsim.geometry.planning.junction_planning.contact import (
    junction_source_binding as junction_source_binding,
)
from scgsim.geometry.planning.junction_planning.core import (
    DEFAULT_FORMAT as DEFAULT_FORMAT,
)
from scgsim.geometry.planning.junction_planning.core import (
    PRECISION_UM as PRECISION_UM,
)
from scgsim.geometry.planning.junction_planning.core import (
    JunctionPartitionFormat as JunctionPartitionFormat,
)
from scgsim.geometry.planning.junction_planning.core import (
    canonical_sha256 as canonical_sha256,
)
from scgsim.geometry.planning.junction_planning.core import (
    plain_record as plain_record,
)
from scgsim.geometry.planning.junction_planning.derivation import (
    derive_rectangular_junctions as derive_rectangular_junctions,
)
from scgsim.geometry.planning.junction_planning.exact_geometry import (
    closed_contact as closed_contact,
)
from scgsim.geometry.planning.junction_planning.exact_geometry import (
    cross as cross,
)
from scgsim.geometry.planning.junction_planning.exact_geometry import (
    inside_ring as inside_ring,
)
from scgsim.geometry.planning.junction_planning.exact_geometry import (
    section as section,
)
from scgsim.geometry.planning.junction_planning.partition import (
    partition_junctions as partition_junctions,
)
from scgsim.geometry.planning.junction_planning.terminal import (
    JunctionTerminalGeometry as JunctionTerminalGeometry,
)
from scgsim.geometry.planning.junction_planning.terminal import (
    junction_terminal_line as junction_terminal_line,
)
