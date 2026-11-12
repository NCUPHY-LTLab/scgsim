"""Expose solution-domain occupancy, adjacency and boundary planning."""

from scgsim.geometry.planning.domain_planning.boundaries import (
    plan_substrate_air_surfaces as plan_substrate_air_surfaces,
)
from scgsim.geometry.planning.domain_planning.envelope import (
    auto_vacuum_envelope_bounds as auto_vacuum_envelope_bounds,
)
from scgsim.geometry.planning.domain_planning.faces import (
    ConductorFaceRequest as ConductorFaceRequest,
)
from scgsim.geometry.planning.domain_planning.faces import (
    conductor_face_solution_pieces as conductor_face_solution_pieces,
)
from scgsim.geometry.planning.domain_planning.faces import (
    interface_surface_kinds as interface_surface_kinds,
)
from scgsim.geometry.planning.domain_planning.faces import (
    planar_side_solution_regions as planar_side_solution_regions,
)
from scgsim.geometry.planning.domain_planning.faces import (
    route_a_sheet_boundary_volume_ids as route_a_sheet_boundary_volume_ids,
)
from scgsim.geometry.planning.domain_planning.faces import (
    route_a_sheet_boundary_volume_ids_from_solutions as route_a_sheet_boundary_volume_ids_from_solutions,
)
from scgsim.geometry.planning.domain_planning.faces import (
    route_a_sheet_plane_z_um as route_a_sheet_plane_z_um,
)
from scgsim.geometry.planning.domain_planning.faces import (
    route_a_sheet_plane_z_um_from_solutions as route_a_sheet_plane_z_um_from_solutions,
)
from scgsim.geometry.planning.domain_planning.hosts import (
    adjacent_auto_vacuum_sidewall_component_id as adjacent_auto_vacuum_sidewall_component_id,
)
from scgsim.geometry.planning.domain_planning.hosts import (
    conductor_sidewall_adjacent_solution_id as conductor_sidewall_adjacent_solution_id,
)
from scgsim.geometry.planning.domain_planning.hosts import (
    conductor_sidewall_ref_solution_id as conductor_sidewall_ref_solution_id,
)
from scgsim.geometry.planning.domain_planning.hosts import (
    sidewall_is_auto_vacuum_envelope_edge as sidewall_is_auto_vacuum_envelope_edge,
)
from scgsim.geometry.planning.domain_planning.occupancy import (
    active_route_conductor_entities as active_route_conductor_entities,
)
from scgsim.geometry.planning.domain_planning.occupancy import (
    entity_loop_bounds as entity_loop_bounds,
)
from scgsim.geometry.planning.domain_planning.occupancy import (
    entity_occupied_region as entity_occupied_region,
)
from scgsim.geometry.planning.domain_planning.occupancy import (
    required_host_solution_id as required_host_solution_id,
)
from scgsim.geometry.planning.domain_planning.occupancy import (
    route_conductor_contact_footprints as route_conductor_contact_footprints,
)
from scgsim.geometry.planning.domain_planning.occupancy import (
    sidewall_on_solution_outer_boundary as sidewall_on_solution_outer_boundary,
)
from scgsim.geometry.planning.domain_planning.occupancy import (
    solution_bounds as solution_bounds,
)
from scgsim.geometry.planning.domain_planning.occupancy import (
    solution_entity_xy_region as solution_entity_xy_region,
)
from scgsim.geometry.planning.domain_planning.occupancy import (
    solution_region_entities as solution_region_entities,
)
from scgsim.geometry.planning.domain_planning.reconciliation import (
    merge_solution_sidewall_interfaces as merge_solution_sidewall_interfaces,
)
from scgsim.geometry.planning.domain_planning.reconciliation import (
    reconcile_solution_domain_boundaries as reconcile_solution_domain_boundaries,
)
from scgsim.geometry.planning.domain_planning.vacuum import (
    prepare_auto_vacuum_solution_regions as prepare_auto_vacuum_solution_regions,
)
