"""Expose conductor interface, contact and ownership planning."""

from scgsim.geometry.planning.contact_planning.intents import (
    complete_interface_plan_from_surfaces as complete_interface_plan_from_surfaces,
)
from scgsim.geometry.planning.contact_planning.intents import (
    recognize_route_interfaces as recognize_route_interfaces,
)
from scgsim.geometry.planning.contact_planning.membership import (
    plan_mm_contact_records as plan_mm_contact_records,
)
from scgsim.geometry.planning.contact_planning.membership import (
    plan_mm_contact_records_with_components as plan_mm_contact_records_with_components,
)
from scgsim.geometry.planning.contact_planning.overlap import (
    is_fully_normalized_face_metal as is_fully_normalized_face_metal,
)
from scgsim.geometry.planning.contact_planning.overlap import (
    validate_volumetric_overlap_ownership as validate_volumetric_overlap_ownership,
)
from scgsim.geometry.planning.contact_planning.patches import (
    plan_conductor_contact_patches as plan_conductor_contact_patches,
)
