"""The production sidecar ``scgsim.sgb.sidecar.v1`` (design brief sections 7 and 12, decision D8).

The sidecar is the JSON companion of the XAO/BREP.  It binds every planned id to its native tag *by
construction order* (tag = 1 + position in ``ids_by_tag``, the ``TopExp::MapShapes`` pre-order of the
written compound -- never by geometry), lists the primary physical groups with the semantic facts
a solver adapter needs, and records everything the file formats cannot carry: embedding, lineage,
inset families, the load-side expectations.  It is deterministic: a fixed key order, no timestamps,
no paths, numbers rounded to twelve significant digits, identical for any worker count and tile layout.
``sidecar.schema.json`` (next to this file) is the contract; tests validate every emitted sidecar
against it.

Top-level fields and who uses them
----------------------------------
``schema`` / ``route`` / ``route_policy`` / ``grid_um`` / ``tolerance_um`` / ``model``
    version and provenance: the route (A, B, C), whether conductor volumes are excluded or
    retained, the model grid (Palace manifest ``curved_arrangement`` provenance is ``arrangement``),
    the identity digest of the model.
``files``
    file names relative to the sidecar (``xao``, ``brep``, ``sliver_report``, ``validation_report``).
``binding.ids_by_tag`` / ``counts``
    planned id <-> native tag per dimension ("0" vertices, "1" edges, "2" faces, "3" volumes).
    Palace: ``live_physical_groups`` replacement and ``entity_tags``; AEDT: the face order of
    each solid is the AEDT face-id order (guarded by counts and area/centre checks).
``physical_groups``
    each primary group by name: ``dim``, ``tag`` (physical tag; not identity, consumers store names),
    ``ids``, ``native_tags``, and for surfaces ``interface_type``, ``entities`` (owner ids),
    ``net``, ``face_kinds``, ``adjacent_domains`` (solution regions: kind, material),
    ``adjacent_volumes``, ``band`` and ``area_um2``; for volumes ``kind`` (vacuum, dielectric,
    conductor -- Route C conductor interiors are ``VOL::<entity>`` of kind conductor) and ``material``.
    Palace rebuilds ``groups = {volumes, boundary_surfaces, surface_groups, pec_surfaces,
    port_surfaces}`` from these: ``owner_semantic_ids`` = ``entities``, ``adjacent_solution_volume_ids``
    = ``adjacent_domains``, ``face_kind`` = ``face_kinds``, ``conductor_component_id`` / ``net_id`` =
    ``entities`` / ``net``, PEC surfaces = the ``MA``/``MS``/``SHEET`` groups of a conductor.
``unions``
    ``interface_types`` (``MA`` -> groups ...; reports aggregate by interface type) and ``nets``
    (Net -> exposed conductor groups; Electrostatic terminals and grounds).  Lists of primary groups,
    never overlapping Gmsh groups.
``surface_epr_rows``
    every Surface-EPR row (``MA::pad::band=2`` ...) and the group that carries it (a Route A sheet
    group carries both its MA and MS row), with entities and band: Palace's cumulative band selection.
``volumes`` / ``faces``
    per planned volume: group, kind, material, Net, ``emitted``, planned volume and ``aedt_name``;
    per planned face: group, ``face_kind``, ``z_um``, the volumes on its two sides.  AEDT's contribution
    ledger (surface id, owner, side, classification MA/MS/SA/MM, normal hint) is read from
    ``faces`` + ``physical_groups`` + ``lineage``.
``embedded_faces`` / ``refinement_edges``
    the (face, host volume) and (edge, host face) pairs the mesher re-applies with
    ``gmsh.model.mesh.embed`` -- XAO cannot carry them (Palace ``embed_route_b_port_surfaces``).
``ports``
    lumped-port sheets: index, name, direction, hosts, layers, group, faces, host volumes.
``lineage``
    edge -> carriers -> source polygon edge or family member; face -> source Entities; membership.
``insets`` / ``refinements`` / ``slivers`` / ``arrangement``
    inset families (request id, support, margins, band -> groups), refinement requests, the sliver
    report summary and the measured snap drift / arc deviation.
``expected``
    the expectations of ``verify_loaded`` (positions, boundaries, areas, volumes).
"""
