"""The production sidecar ``scgsim.sgb.sidecar.v5`` (design brief sections 7 and 12, decision D8).

The sidecar is the JSON companion of the XAO/BREP and SGB's complete record of the model: everything the build
used, and everything the mesher, the meshing-stage check and the solver adapters need, so a reader reads and
never reconstructs what SGB knew.  It binds every planned id to its native tag *by construction order*
(tag = 1 + position in ``ids_by_tag``, the ``TopExp::MapShapes`` pre-order of the written compound -- never by
geometry).  It is deterministic: a fixed key order, one line per record, no timestamps, no paths, numbers
rounded to twelve significant digits, identical for any worker count and tile layout.  The contract is one
definition, ``scgsim.contracts.sgb_sidecar``: the records (``records``), the version, the shared reader that
checks version, structure and every reference, and the views readers share (``views``).
``sidecar.schema.json`` (next to this file) is the JSON Schema derived from it by
``scripts/generate_sidecar_schema.py``.

Top-level fields
----------------
``schema`` / ``model`` / ``binding`` / ``summary``
    the version; the model's name, identity digest, route (A, B, C) and route policy, grid, tolerance and the
    file names beside the sidecar; planned id <-> native tag per dimension ("0" vertices, "1" edges, "2" faces,
    "3" volumes: Palace's ``live_physical_groups`` replacement and ``entity_tags``; AEDT's face order of each
    solid); the counts of the written file, the sliver report's summary (with the snap drift and arc deviation)
    and the number of arc edges.

Every other field is a table ``{id: record}``.  Records name records of other tables by id, typed in the
contract, and the shared reader checks that each resolves.  Ids are opaque: SGB builds them, and no reader
parses one; every fact a reader needs is a field.

Sources -- what the layout drew and asked for
---------------------------------------------
``entity``
    each Entity's prisms (one per polygon): kind (conductor, dielectric, vacuum, port), material, Net, the source
    layer it was drawn on, z range and feature kind.
``source_curve``
    each straight edge and circular arc of each Entity outline: its Entity, polygon, ring and edge number, its
    exact curve (line ends, or arc centre, radius, sense and ends) and length.
``inset_request``
    the inset requests as given.  A face request (``scope`` face) names one face and is its own band family; a
    type-level request (``scope`` type: every face of a Surface-EPR type, optionally only some face kinds,
    without the excluded Entities) lists the ``families`` the physical pre-pass created, in face order, and the
    faces it ``skipped`` with the reason (``no_gap_edge``).  A port can cover a family's whole face, so whether
    a family has bands to integrate is read from its own ``bands`` and from ``surface_epr_row``.
``port``
    each lumped-port sheet: index, name, direction, its Entity and hosts, layers, group, faces, host and
    adjacent volumes.

Derived -- what SGB computed from the sources
---------------------------------------------
``cut_family``
    every cut family (the inset bands, and any other role a catalogue gives): its role and every level its
    curves are drawn at; a family a carrier names that the catalogue lacks is listed with neither.  Every
    curve of a family is a cell boundary.
``band_family``
    one per banded face: its face, margins (also in model-grid units, the identity a margin is looked up by),
    band count, support area, band index -> the surface groups of its faces (each a key of ``group``), and band
    index -> the metal-metal contacts in that band that no group carries (``hidden_contacts``: Route B writes no
    conductor volume, so a contact between two conductors is no written face).
``support_ring`` / ``support_primitive``
    the exact boundary of each family's support region: each ring's primitives in order (their count is the
    ring's size), and per primitive its curve, length, whether bands are measured from it (a distance origin),
    whether it lies on the die edge, and the source curves drawn along it.
``carrier``
    every carrier an emitted horizontal edge lies on: a source curve (or a witness chord of one), a band curve
    with the support primitive it follows, its role (edge, corner, capstart, capend), margin and piece, or a
    member of another cut family (its family and member name).

Emitted -- the shapes that entered the model
--------------------------------------------
``vertex`` / ``edge`` / ``face`` / ``volume``
    every written vertex's position; every written edge's kind (horizontal or vertical), vertices, arc, length
    and carriers (or, for a wall column, the Entities whose walls stand on it); every written face's group,
    kind, face kind, z range, side volumes, written wires, planned area, whether it is curved, what the curves
    on its boundary come from (``entities``, ``band_families``, ``other_cut_families``: classified by the
    carrier record, never by id), the fraction of its area each Entity covers (``entity_coverage``,
    non-exclusive) and the bands it lies in (``bands``); every planned volume's Entity and domain,
    ``emitted``, group, shells, planned volume, ``aedt_name``, the faces of its written solid and the sheets
    embedded in it.  The load-side expectations by native tag (``views.expected``, which ``verify_loaded`` and
    the OCCT and STEP checks compare) and the sheets the mesher re-embeds with ``gmsh.model.mesh.embed``
    (``views.embedded_faces``) are computed from these.
``group``
    each primary group by name: ``dim``, ``tag`` (physical tag; not identity, consumers store names), and for
    surfaces ``interface_type``, ``entities`` (owner ids), ``net``, ``face_kind``, ``adjacent_domains``
    (solution regions: kind, material), ``adjacent_volumes``, ``band`` and ``area_um2``; for volumes ``kind``
    (vacuum, dielectric, conductor -- Route C conductor interiors are the volume group ``<entity>`` of kind
    conductor) and ``material``.  A face or volume names its group; the members are computed
    (``views.group_members``).  A volume group, and a Route A conductor sheet group, is named by the bare
    entity id (the Gmsh Type column gives the dimension); ``interface_type`` of a sheet group is ``SHEET`` -- a
    field derived from the face structure, not part of the name.  Palace rebuilds ``groups = {volumes,
    boundary_surfaces, surface_groups, pec_surfaces, port_surfaces}`` from these: ``owner_semantic_ids`` =
    ``entities``, ``adjacent_solution_volume_ids`` = ``adjacent_domains``, ``face_kind`` = ``face_kind`` (one
    per group, and part of its name: ``<kind>::<entity>__<top|bottom|sidewall|sheet|interface>[::band=k]`` for a
    prefixed group, where, when a band group of an ``interface`` family is shared with another family, every
    band group of both families gains the side of the dielectric its faces lie on
    (``...__interface__top::band=k`` and ``...__interface__bottom::band=k``, still ``face_kind``
    ``interface``); a sheet group ``<entity>[::band=k]`` has face kind ``sheet`` without carrying it in the
    name), ``conductor_component_id`` / ``net_id`` = ``entities`` / ``net``, PEC surfaces = the groups of
    interface type ``MA``/``MS``/``SHEET`` of a conductor.
``surface_epr_row``
    every Surface-EPR row (``MA::pad__top::band=2`` ...) and the group that carries it (whose ``entities``
    are the row's), with its band: Palace's cumulative band selection.  A band row names the surface it is a band of
    (``banded_surface``: ``MA::pad__top``, the row's name without its band), so the bands of one surface are
    found by a field, never by cutting a row id; an unbanded row has ``None``.  A Route A sheet group (``pad::band=2``) carries
    ``MA::pad__sheet::band=2`` when a face of it has vacuum on some side and ``MS::pad__sheet::band=2``
    when a face has a dielectric on some side; a sheet with vacuum on both sides has no ``MS`` row.  A sheet
    row has ``shared_face_group`` (the sheet group), ``face_sides`` (face id -> ``above``, ``below`` or
    ``both``, only the faces that have that neighbour) and ``side`` (their common value, or ``mixed``), so
    the consumer decides internal-boundary cracking from the row alone; every other row has ``None``
    for the three.
``mesh_feature``
    facts about the geometry for the mesher, never a size: ``film_sidewall`` (the sidewall outline of a
    finite body, with its ``thickness_um``), ``band_boundary`` (a band family's edges, with the request's
    ``margins_um`` and ``narrowest_band_um``), ``arc`` (the circular-arc edges of one ``radius_um``),
    and ``port_edge``; each lists edge ids and the Entity, family or port it comes from.  The mesher decides every size (see ``features``).

AEDT's contribution ledger (surface id, owner, side, classification MA/MS/SA/MM, normal hint) is read from
``face`` and ``group``.  The meshing stage of ``simulation_drc`` (``scgsim.meshing.check_fine_structure``)
reads the load-side view as the geometry of the built model, and ``edge``, ``face``, ``carrier``, the support
tables and ``entity`` for where each fine structure comes from and the layer it was drawn on.
"""
