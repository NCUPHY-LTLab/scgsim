"""The production sidecar ``scgsim.sgb.sidecar.v2`` (design brief sections 7 and 12, decision D8).

The sidecar is the JSON companion of the XAO/BREP.  It binds every planned id to its native tag *by
construction order* (tag = 1 + position in ``ids_by_tag``, the ``TopExp::MapShapes`` pre-order of the
written compound -- never by geometry), lists the primary physical groups with the semantic facts
a solver adapter needs, and records everything the file formats cannot carry: embedding, lineage,
inset families, the load-side expectations.  It is deterministic: a fixed key order, no timestamps,
no paths, numbers rounded to twelve significant digits, identical for any worker count and tile layout.
The contract is one definition, ``scgsim.contracts.sgb_sidecar``, which also holds the version constant
and the shared reader; ``sidecar.schema.json`` (next to this file) is the JSON Schema derived from it by
``scripts/generate_sidecar_schema.py``, and tests validate every emitted sidecar against it.

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
    ``net``, ``face_kind``, ``adjacent_domains`` (solution regions: kind, material),
    ``adjacent_volumes``, ``band`` and ``area_um2``; for volumes ``kind`` (vacuum, dielectric,
    conductor -- Route C conductor interiors are the volume group ``<entity>`` of kind conductor) and
    ``material``.  A volume group, and a Route A conductor sheet group, is named by the bare entity id
    (the Gmsh Type column gives the dimension); ``interface_type`` of a sheet group is ``SHEET`` -- a
    field derived from the face structure, not part of the name.
    Palace rebuilds ``groups = {volumes, boundary_surfaces, surface_groups, pec_surfaces,
    port_surfaces}`` from these: ``owner_semantic_ids`` = ``entities``, ``adjacent_solution_volume_ids``
    = ``adjacent_domains``, ``face_kind`` = ``face_kind`` (one per group, and part of its name:
    ``<kind>::<entity>__<top|bottom|sidewall|sheet|interface>[::band=k]`` for a prefixed group, where, when a band group
    of an ``interface`` family is shared with another family, every band group of both families gains the side of the dielectric
    its faces lie on
    (``...__interface__top::band=k`` and ``...__interface__bottom::band=k``, still ``face_kind`` ``interface``);
    a sheet group ``<entity>[::band=k]`` has face kind ``sheet`` without carrying it in the name),
    ``conductor_component_id`` / ``net_id`` = ``entities`` / ``net``, PEC surfaces = the groups of interface type ``MA``/``MS``/``SHEET`` of a conductor.
``unions``
    ``interface_types`` (``MA`` -> groups ...; reports aggregate by interface type; a Route A ``SHEET``
    group is listed under ``SHEET``, and under ``MA`` / ``MS`` when it carries that row) and ``nets``
    (Net -> exposed conductor groups; Electrostatic terminals and grounds).  Lists of primary groups,
    never overlapping Gmsh groups.
``surface_epr_rows``
    every Surface-EPR row (``MA::pad__top::band=2`` ...) and the group that carries it, with entities and
    band: Palace's cumulative band selection.  A Route A sheet group (``pad::band=2``) carries
    ``MA::pad__sheet::band=2`` when a face of it has vacuum on some side and ``MS::pad__sheet::band=2``
    when a face has a dielectric on some side; a sheet with vacuum on both sides has no ``MS`` row.  A sheet
    row has ``shared_face_group`` (the sheet group), ``face_sides`` (face id -> ``above``, ``below`` or
    ``both``, only the faces that have that neighbour) and ``side`` (their common value, or ``mixed``), so
    the consumer decides internal-boundary cracking from the row alone; every other row has ``None``
    for the three.
``mesh_features``
    facts about the geometry for the mesher, never a size: ``film_sidewall`` (the sidewall outline of a
    finite body, with its ``thickness_um``), ``band_boundary`` (an inset family's edges, with the request's
    ``margins_um`` and ``narrowest_band_um``), ``arc`` (the circular-arc edges of one ``radius_um``),
    ``refinement_curve`` and ``port_edge``; each entry lists edge ids and native tags and the request, entity
    or port it comes from.  The mesher decides every size (see ``features``).
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
    the inset requests as given, refinement requests, the sliver report summary and the measured snap
    drift / arc deviation.  A concrete inset request is the record of its face (request id, support,
    margins, band -> groups; every request on that face asks the same margins).  A type-level
    request (every face of a Surface-EPR type) has the request as given (type, face kinds, excluded
    entities, margins) and its ``expansion``: ``banded``, the record of each face it banded (ids
    ``<request id>::<entity>::<side>``), and ``skipped``, each face left unbanded with the reason
    (``no_gap_edge``).  ``banded`` lists the families the physical pre-pass created, before any port is
    placed; whether a family has bands to integrate is read from its own ``bands`` and from
    ``surface_epr_rows``, because a port can cover a family's whole face (``bands`` is then ``{}``).
    A build without a type-level request writes the same bytes as before they existed.
``expected``
    the expectations of ``verify_loaded`` (positions, boundaries, areas, volumes), the load check the tests
    run; the mesher trusts the files of one build and does not check them against it.  The meshing stage of
    ``simulation_drc`` (``scgsim.meshing.check_fine_structure``) reads its positions, boundaries and face
    areas as the geometry of the built model.
"""
