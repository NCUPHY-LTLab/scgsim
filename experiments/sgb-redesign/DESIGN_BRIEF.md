# SGB redesign — architecture brief (authoring source for writers)

Owner: Lead - SCGSim Architecture (Claude Code). This brief is the semantic
authority writers expand into `docs/design/sgb/`. Writers may restructure prose
and add figures/examples; they must not change a decision here. Anything a
writer finds ambiguous or wrong goes back to the Lead as a question.

## 0. One-paragraph summary

A superconducting chip model is a finite set of **prisms**: a planar footprint
extruded over a Z interval (a sheet is a prism of zero thickness). SGB derives
the *complete* conformal solid model from those prisms **before any CAD kernel
is involved**: it builds one 2D arrangement per Z **level** and per **slab**,
labels every planar cell with its material, owner and semantic surface, and
assembles faces, vertical walls, shells and volumes from shared atomic
vertices and edges. Every way of partitioning a surface — material
boundaries, port footprints, contact caps, Surface-EPR inset bands, mesh
refinement curves, parallel tile seams — is the same operation: **add a
family of curves to the arrangement with a semantic role**. The CAD kernel
(OCC via Gmsh) only *emits* the already-decided topology bottom-up, with no
Boolean operation, so cost is linear in the model size and every native
entity has a planned identity. All derivation steps are pure functions over
immutable records, so they parallelise over levels, slabs, spatial tiles,
regions and cells.

## 1. Goals and non-goals

Goals (Human-stated, in priority order):

1. **Semantics before geometry.** Material, owner, Net, interface kind,
   route representation and surface identity are decided on planned records;
   the kernel never decides them.
2. **No global fragment; scale to large geometry.** No 3D Boolean on the
   model. Cost must grow ~linearly (n log n) in feature count. Derivation
   must be parallelisable.
3. **Stable source identity, forward-propagated; route is an early input.**
   What geometry to build is fully decided before construction.
4. **Reproducible handoff identity and programmatic mapping** from every
   solver entity (Palace attribute, Gmsh physical group, XAO entity) back to
   semantic surfaces/volumes and source curves — never by human or AI
   inspection, never by coordinate/area matching.
5. **Backend neutral.** One closed, conformal **solid** model. Palace
   consumes it through XAO→Gmsh; AEDT consumes the same solids in a later
   stage (interface defined now, implementation later).
6. **Conformal by construction; slivers eliminated where they are artefacts
   and reported where they are source facts.**
7. **Extensible partitioning** (Section 5): inset imprint is the first
   consumer, not a special case.

Non-goals (v1): meshing algorithms (Gmsh owns them; SGB supplies geometry and
size hints only), solver configuration, inferring electrical connectivity,
non-prismatic bodies (sloped sidewalls, curved-in-Z bridges). Non-prismatic
bodies are a documented extension point, not silently approximated.

## 2. The 2.5D model: prisms, levels, slabs

- **Prism** `P = (footprint F, [z0, z1], material, entity_id, role)`.
  Footprint = planar region bounded by curve loops (outer + holes). `z0 = z1`
  is a sheet (Route-A face metal, lumped-port sheet).
- **Levels** `Z = sorted distinct {z0, z1}` of all prisms (plus sheet planes,
  port planes). **Slab** `s_i = (z_i, z_{i+1})`.
- **Active set** of slab i: prisms with `z0 <= z_i` and `z1 >= z_{i+1}`.
  Active sheets at level i: sheets with `z0 = z1 = z_i`.
- **Slab arrangement** `S_i`: planar subdivision of the XY plane by the
  boundary curves of the footprints active in slab i. Each cell gets a slab
  label: the occupying material/domain (resolved by the existing explicit
  priority/ownership rules; different-Net metal overlap is an error).
- **Level arrangement** `A_i`: planar subdivision by the curves of
  `S_{i-1} ∪ S_i ∪ sheets at z_i ∪ cut-family curves targeting level i`.
  Each cell gets a level label `(below_domain, above_domain, sheet_owner?,
  family_bands…)`.
- **Why local, not one global projection.** Projecting every layer's curves
  onto one global arrangement (as the current curved path does) partitions
  every level by curves of unrelated layers: more faces, more mesh
  constraints, and *phantom slivers* where two unrelated layers have nearly
  coincident edges. Local arrangements only co-arrange curves that actually
  meet in 3D. This is the first sliver rule.

Conformity by construction:
- `A_i ⊇ curves(S_i)` and `A_{i+1} ⊇ curves(S_i)`, so every slab-i edge is a
  chain of atomic edges in both bounding levels.
- **Walls**: for each maximal edge chain of `S_i` whose two sides have
  different slab labels, emit one vertical face per *atomic slab edge*. Its
  bottom boundary is the chain of `A_i` edges covering that slab edge, its top
  boundary the chain of `A_{i+1}` edges, and two vertical edges at the slab
  edge endpoints (shared with neighbouring walls). Walls of an arc carrier
  are cylindrical faces, of a spline carrier ruled faces.
- **Wall coalescing** (decided after prototype): a level only cuts a wall
  where something at that level is incident on it (an edge or vertex of
  `A_i` lying on the wall's horizontal line, e.g. a port sheet edge touching
  a metal sidewall). Otherwise the stacked walls of consecutive slabs merge
  into one face. Levels are global in Z but their effect is local in XY.
- **Level labels** are obtained by linking each level cell to the slab cells
  below and above it (one source of truth), not by a second point test.
- **Horizontal faces**: connected components of `A_i` cells after edge
  dissolution (Section 4) with identical label.
- **Volumes**: connected components of slab cells with the same domain
  identity, connected within a slab through shared non-wall edges and across
  slabs through level cells whose below/above domains are equal (no face
  emitted there). Each volume's shell = its horizontal faces + walls, all
  shared with neighbours. Holes in shells (inner PEC voids for Route B) are
  just additional closed shells of the same volume.

## 3. Kernel and precision: snap-rounded arrangement with exact carriers

- One **grid** `g` per model, default = source DBU (1 nm). Declared once,
  recorded in the handoff. Z values are not snapped by the XY grid.
- **Carriers**: every curve keeps its exact definition (line through two
  points, circle (centre, radius), source B-spline). Arrangement vertices
  are **snapped to the grid** (snap rounding: Hobby; Guibas–Marimont;
  Halperin–Packer iterated snap rounding), giving: all vertices on grid,
  no two vertices closer than `g`, no edge shorter than `g`, edges stay
  within `√2·g/2` of their carriers, topology consistent (no crossing edges).
- **Pixel convention** (decided after prototype): half-open pixels centred on
  the grid point, `[c−g/2, c+g/2)²` (lower-left boundary included); a segment
  touching only a pixel corner does not make it hot.
- **Deviation bound** (decided after prototype): one snap-rounding round
  keeps every output vertex within `√2·g/2` of its carriers; iterated snap
  rounding adds vertex–edge separation but can drift further (measured up to
  1.93 g on dense adversarial random segments, 0.605 g on a random layout with N=200 (the earlier 0.61–0.70 g has no reproducible source and is dropped)).
  Layout sources are already on the grid, so drift only arises at crossings
  of oblique carriers. The arrangement records the measured per-edge
  deviation; a drift cap (bounded-drift ISR, Packer) is open decision D7.
- Geometry realisation: a line edge is the segment between snapped
  vertices; an arc edge is the circle arc through the two snapped endpoints
  with the carrier's centre-side and a radius chosen to pass through them
  (deviation bounded by the snap bound). Recorded per edge: carrier id,
  carrier parameter interval, max deviation.
- Consequence: **integer topology is exact and order-independent**, which is
  what makes tile-parallel merging deterministic (Section 6). Floating-point
  tolerances never decide topology.
- Curve classes in v1: lines and circular arcs (closed under offset, so
  inset curves stay in-class). Source B-splines are allowed as carriers for
  material boundaries; offsets of B-splines are not exactly representable
  and are **rejected for inset in v1** (approximation would need a Human
  tolerance).
- Implementation note (for the plan, not the doc's promise): rectilinear and
  polygonal work can use KLayout's integer Region engine (C++, scanline,
  `TilingProcessor` multithreading) for noding and labelling; arcs need our
  own snap-rounding over line/arc carriers. The experiments decide whether
  arcs are noded natively or through a bounded-deviation polyline witness
  with carrier back-mapping. (Open decision D3.)

## 4. Labels, required edges and dissolution

- After noding, an edge is **required** iff the labels on its two sides
  differ, or it belongs to a curve family whose role is `required` (e.g. an
  inset band boundary, a port footprint, a refinement curve).
- Non-required edges are **dissolved**: their two cells merge. This removes
  same-material overlap seams, tile seams, and degree-2 vertices on the same
  carrier. Dissolution is what keeps the model minimal; a seam never
  survives unless something semantic needs it.
- Label granularity is set by the **attribution policy** (route-owned):
  default `(interface kind, Net, material pair, family band)`. Per-Entity
  lineage on a merged face is recorded as area-weighted membership, not as a
  seam (an Entity is lineage, not a promise of a per-Entity measurement).

## 5. Cut families — the extension mechanism

A **cut family** is an immutable record:
`(family_id, role, generator, target selector, edge policy, distance origin)`.

| Family | Generator | Target | Edge policy |
|---|---|---|---|
| physical boundaries | prism footprints | levels/slabs where active | required iff labels differ |
| lumped port footprint | port sheet polygon | its plane | required |
| contact caps | (automatic: overlap of prisms) | contact level | required iff labels differ |
| Surface-EPR inset bands | offset of a *physical support region* by margins m_1<…<m_k | selected semantic surfaces | required; band index joins the label |
| mesh refinement curves | user/automatic | any surface | `refinement_only`: kept as edges, not a label change |
| tile seams | parallel scheduler | all levels | always dissolved |

Adding a new partition kind = writing a generator + selecting a target +
choosing an edge policy. No new code path in assembly or emission.

**Distance-origin rule** (fixes a defect of solver-side masks): inset curves
are generated from the *physical boundary of the support region*, computed
before arrangement and before any partition. Seams created by other families
(ports, contact caps, tile seams, other bands) are never distance origins.

## 6. Inset imprint (first consumer)

- Inset region `R_m = R ⊖ D(m)` (Minkowski erosion by a disk). Exact boundary:
  each edge offset inward by m; **convex corners stay sharp** (offset lines
  meet); **reflex corners become arcs of radius m** centred at the reflex
  vertex; a circular hole of radius r becomes a concentric circle r+m.
  Built from the offset arrangement of edges/arcs, then keep the
  `dist ≥ m` cells (no polygonised disk, no 128-gon).
- Bands: `B_0 = R \ R_{m1}`, `B_k = R_{mk} \ R_{mk+1}`, `B_K = R_{mK}`.
  Each band face carries `(contribution, margin ordinal k)`. The quantity
  "at least m_k from the edge" is the union of bands ≥ k — selections are
  cumulative and must not be added as disjoint contributions.
- Topology changes (erosion splits a neck, a band vanishes) are ordinary
  outcomes, recorded per band.
- Decided after prototype phase 2:
  - "Exact" means **exact carriers, snapped end points**: an offset of an
    oblique edge has irrational end points; carriers stay exact, vertices
    are on the grid, measured offset-arc deviation ≤ 0.63 g.
  - Circles are carried as four quarter arcs (a full circle is a loop edge
    and a half circle is parallel to its chord); reflex arcs split at 90°.
  - Bands are generated per support face: Route A sheet (one face carrying
    both the metal–vacuum and metal–substrate rows), Route B top face (MA)
    **and bottom face (MS)** separately; sidewall bands are out of v1 scope.
  - Route A sheet faces are named by the face (`SHEET::<net>::band=k`); the
    MA and MS Surface-EPR rows both reference that one group.
  - Curved walls: `addBSplineFilling` through four shared edges reproduces
    the cylinder exactly (deviation 1e-11 g); walls with more boundary edges
    need direct BREP writing (D8) or wall splitting. `addSurfaceFilling`
    is not exact (0.03–0.28 µm) and is not used.
  - Arc centre points are construction aids, removed before meshing.
- Palace consumption: each band is its own physical group / attribute set;
  the masked integral becomes a plain integral over the union of band groups —
  conformal with the mesh, no indicator cutting through elements, so AMR
  refines the true band boundaries and masked quantities converge with the
  field discretisation only. Stock (upstream) Palace only needs plain surface
  groups for this. The forked Palace with a native `Mask: Inset` is NOT used
  and NOT supported by this design: removing the need for it (its approach
  has performance problems) is a goal of the redesign — inset is solved in
  CAD/XAO (Human direction).
- AEDT consumption (later): the same band faces exist on the exported solids
  (imprinted), replacing non-model integration sheets.

## 7. Identity and the mapping chain

Two identity layers:
- **Semantic identity** (stable across unrelated edits): physical names from
  labels, e.g. `MA::Net=xmon_pad::band=2`, `SA::substrate|vacuum::band=0`,
  `VOL::vacuum_outer`. Used for physical groups, Surface-EPR rows and reports.
- **Atomic topology identity** (reproducible, content-addressed): vertex =
  H(level, grid coordinate); edge = H(carrier id, endpoint ids, level|slab);
  face = H(label, sorted boundary edge ids); volume = H(domain, cell set).
  Same input ⇒ same IDs regardless of worker count or tile layout.
- **Lineage**: every edge → carrier → (source entity, polygon, ring, edge
  index | family member); every face → contributing source entities (area
  fractions) and family bands.
- Mapping chain (all programmatic): Palace attribute → Gmsh physical group →
  semantic surface key → atomic faces → carriers → source entity/curve.
  Native tags are opaque and only recorded, never used to infer identity.
- **Physical groups partition each dimension** (decided after prototype):
  every emitted surface/volume belongs to exactly one primary physical
  group. Unions such as "all PEC faces of Net X" are lists of primary
  groups in the sidecar and solver config, never overlapping Gmsh groups
  (MSH output duplicates elements per group, which solvers can misread).
- Embedded sheets that bound no volume (Route-B mid-plane lumped port) are
  emitted as faces embedded in their host volume; XAO cannot carry the
  embedding, so the sidecar records `(face, host volume)` and the mesher
  re-applies it.
  Emission binds native tags to planned IDs by construction order, not by
  geometry matching.

## 8. Pipeline (stages and records)

| Stage | Input → output | Parallel unit |
|---|---|---|
| S0 Source snapshot | GeometryPlan → normalized source | occurrence |
| S1 Physical interpretation | source → effective prisms (existing contract) | entity |
| S2 Prism catalogue | effective input → `PrismRecord`s, levels, slabs | entity |
| S3 Cut families | prisms + requests → `CurveFamilyRecord`s (e.g. offsets) | region × margin |
| S4 Arrangement | curves → snap-rounded `A_i`, `S_i` | level/slab × tile |
| S5 Labelling & dissolution | arrangements + prisms → labelled minimal cells | cell |
| S6 Topology assembly | cells → vertices/edges/faces/walls/shells/volumes | slab, volume |
| S7 Route representation | topology → emitted/excluded volumes, PEC faces | face/volume (label rule only) |
| S8 Identity & groups | topology → semantic names, physical-group plan, lineage | face |
| S9 Validation | all records → invariant report, sliver report | independent checks |
| S10 Emission | plan → OCC bottom-up → XAO (+ later solids for AEDT) | sequential (linear) |

Route A/B is a labelling policy in S7 (and a prism interpretation in S1),
never a different geometry pipeline. Route C (retained conductor volumes):
open decision D1.

## 9. Parallelism

- Derivation (S1–S9) is the parallel target; Gmsh meshing is already
  natively parallel and out of scope.
- DAG: S2 → {S3 per region} → {S4 per level/slab, per tile} → {S5 per cell}
  → {S6 per slab} → S7/S8 (per face) → S9 → S10.
- Spatial tiling within a level (decided after prototype; replaces the
  halo/seam-split idea, which created extra hot pixels and broke identity):
  tile cut lines lie *between pixel columns/rows*, so every hot pixel has
  exactly one owning tile. Each tile receives the whole, unclipped carriers
  that intersect it, snap-rounds them, and keeps only the edge pieces inside
  its pixels; the merge concatenates chain pieces in order. No seam geometry
  ever enters the arrangement, so the merged result is identical to the
  single-tile result by construction (property-tested: same vertex, edge and
  face ids).
- **Granularity** (decided after prototype): tiling only the noding step
  gave no speed-up, because noding was ~15 % of derivation and the
  remaining stages were serial. The parallel unit is therefore the whole
  per-tile derivation — noding, face extraction, labelling and atomic id
  hashing — per (level | slab) × tile; only shell/volume connectivity is a
  global reduction over tile summaries.
- Workers are processes (pure functions over picklable records). Gmsh/OCC
  state is never shared across workers; S10 runs in one process and is
  linear in entity count.
- Report: measured speedups come from experiment E1; the design states the
  model (Amdahl split between parallel derivation and sequential emission).

## 10. Slivers

1. Quantization slivers: eliminated by snap rounding (no vertex pair < g,
   no edge < g).
2. Phantom slivers from unrelated layers: eliminated by local arrangements.
3. Seam slivers: eliminated by dissolution.
4. Source near-coincidence (two genuinely different boundaries a few nm
   apart): source fact. Default = keep and **report** (location, width,
   carriers, entities). Optional Human-authorised policies: a coarser snap
   grid per layer/family; **derived-to-source snapping** — derived curves
   (inset, refinement) may snap onto source curves within τ_derived, source
   curves never move. Every threshold is a Human Gate (D2).

## 11. Validation invariants (structural, not research Gates)

Each face used by ≤ 2 volumes (exactly 2 for interior faces); each shell
closed and orientable (every edge used twice with opposite orientation within
a shell); Euler characteristic consistent per shell; total area per level
equals arrangement area; per-prism footprint area conserved across its
cells; no edge shorter than g; every emitted native entity has exactly one
planned ID; every physical group non-empty; band areas non-increasing with k.
Failures stop the build (fail closed); no fallback geometry.

## 12. Emission

(Superseded by D8: emission serialises the same hierarchy directly to
BREP/XAO; the gmsh-API description below is the test oracle.)
OCC via Gmsh API, strictly bottom-up: addPoint (once per vertex) → addLine /
addCircleArc / addBSpline (once per atomic edge) → addCurveLoop →
addPlaneSurface / wall surfaces (addSurfaceFilling or ruled through shared
edges) → addSurfaceLoop → addVolume. No fragment/cut/fuse/heal; OCCAutoFix
off (a needed fix is a planning defect). Physical groups from S8. Write XAO
plus a JSON sidecar (planned ID ↔ native tag in emission order). Size hints
(refinement curves, band edges) exported for the Palace mesher.
AEDT: later stage emits per-material solids with shared faces (format D4).

## 13. Open decisions (for Human / Codex Architecture review)

- D1 Route C: drop or support as a labelling policy.
- D2 Sliver thresholds and derived-to-source snapping τ (Human Gate).
- D3 Arc noding: native snap rounding over arcs vs polyline witness with
  carrier back-mapping.
- D4 AEDT solid exchange format (STEP/SAT/native PyAEDT construction) and
  face-identity transport.
- D5 Attribution granularity defaults (per Net vs per Entity surfaces).
- D3 provisional decision (Lead, pending review): v1 nodes arcs through a
  **polyline witness** (chords with sagitta ≤ g/4) carrying the arc carrier
  id and parameter interval; after snap rounding, consecutive witness pieces
  of one arc carrier with no foreign hot pixel between them merge back into
  one arc edge realised from the carrier through its snapped endpoints.
  Structural validation is the safety net; native arc snap rounding
  (Eigenwillig et al.) is the upgrade path.
- D8 Emission — Lead decision after E3 (pending review): **direct BREP/XAO
  serialisation** of the planned topology is the emission path. Same
  0D→1D→wire→2D→shell→3D hierarchy, written as an OCCT `CASCADE Topology
  V1` text instead of gmsh API calls. Measured at N=20000: write 75.6 s →
  8.4 s, write+load 117 s → 46 s (Gmsh load ≈ 37 s is the floor for any
  format); exact circles and cylinders; shared faces written once;
  planned id → native tag by construction (`TopExp::MapShapes` pre-order);
  output byte-identical for any worker count. Pcurves are written (required
  for cylinders; planes alone would not need them); tolerance 1e-7 µm.
  The gmsh-API emitter is kept only as a test oracle. STEP for AEDT via
  OCCT works but duplicates shared faces per solid (D4).
- D9 Production kernel language: Python orchestration with a compiled
  noding/DCEL/dissolution kernel (measured Python ceiling ≈ 2.3× from
  parallelism; serial fraction ≈ 45 %).
- D7 Drift cap for iterated snap rounding (bounded-drift ISR) or record-only.
- D6 Replacement plan: both current paths (polygon and curved) are removed
  when the new pipeline lands; public `GeometryPlan` API kept.

## 14. Decisions after the documentation pass (Lead)

Resolving writer questions; these override earlier wording where they differ.

- **Refinement curves** are `refinement_only`, never `required` (§4 example
  corrected). A refinement piece that lands in no emitted face is an error,
  not silently dropped; curve-in-face embeddings are sidecar records.
- **Tile seams are not a cut family.** With between-pixel cuts no seam
  geometry enters the arrangement, so the `dissolve` edge policy is removed.
- **Different-Net contact is an error in every route**, including two
  Route-A sheets sharing an edge or a vertex on one plane, and including a
  contact created by snapping (a coarse grid or derived-to-source snapping
  that closes a source gap between different Nets must fail, naming both
  entities and the location). Same-Net contact is normal.
- **Area-weighted membership** (D5 default): per emitted face, for every
  contributing source entity, the fraction of the face area covered by that
  entity's footprint. Shares are non-exclusive (overlaps count for each
  entity) and can sum to more than 1. Recorded in lineage.
- **Lineage of straight inset members** keeps its generating primitive
  (`<support>/ring<r>/edge<k>`), like arc members.
- **Band-area monotonicity** (`area(R_m)` non-increasing in m) is a
  validation invariant and must be implemented.
- **Sliver width** is the local feature size: the minimum distance between
  two non-incident boundary edges of a face, measured on the exact carriers
  (arcs included), not 2A/P on the chord polygon.
- **Route A sheet band groups** are named `SHEET::<net>::band=k`; the MA and
  MS Surface-EPR rows both reference that group. Route B generates bands on
  the top face (`MA::<net>::band=k`) and bottom face (`MS::<net>::band=k`).
- **Edge id** = H(plane key, sorted endpoint vertex ids, sorted carrier-id
  set) — the prototype's formula is the design.
- **Validation cannot be disabled** in production builds; a test-only switch
  stays in the prototype.
- **Physical tags are not identity.** Tags follow sorted group names and
  shift when groups are added; consumers store group names.
- **AMR wording.** Conformal AMR keeps band edges as element edges, so the
  masked integral has no indicator-cut error; whether refinement
  concentrates near band edges depends on size hints and the error
  estimator. No claim beyond that until a Palace run measures it.
- **Wall coalescing, complete rule set**: a level cuts a wall where (a) an
  edge or vertex of that level lies on the wall's horizontal line, or (b) a
  sheet face of that level lies along the wall chain; a wall column (vertical
  edge) is split wherever a neighbouring wall run starts/ends or a horizontal
  face has a vertex on that column.
- **Cut-family target levels**: a family targeting a Z that no prism
  defines adds that level (it is an ordinary level with no active-set change).
- **Gmsh API cannot express solid sense** (prototype phase 3): even with
  `addSurfaceLoop(sewing=True)` the orientation of a solid is not
  controllable (random N=1000 layout: vacuum inside out); sewing a ~60k-face
  shell took 180 s and 28 GB. This is additional evidence for D8: the direct
  writer states every orientation explicitly.
- **Non-manifold pinch**: finite metal of one Net meeting another body only
  at a vertical edge or a vertex (no shared face) is rejected with a clear
  error; sheets may meet at a corner.
- **Validation layers** (Human-accepted): (1) development tests run OCCT's
  `BRepCheck_Analyzer` and `BOPAlgo_ArgumentAnalyzer` (OCP, dev-only
  dependency) plus Gmsh load checks over broad fixtures and random layouts;
  (2) every production build fails closed on SGB's own structural
  invariants before writing and on load checks in the mesher (counts,
  boundaries, areas, volumes, groups); (3) an optional strict mode runs the
  OCCT checks on production output when OCP is installed. End-user installs
  (e.g. through OrcaPen SC PDK) do not pull OCP.
- **D4/D8 settled by E4** (AEDT 2024.2, PyAEDT 1.3.0; Human's adoption
  condition met). AEDT path = OCCT STEP as an **XDE assembly**, one named
  component per solid (names land as AEDT object names after AEDT's
  sanitisation to `[A-Za-z0-9_]`; SGB writes a sanitised, collision-checked
  `aedt_name` into the sidecar and asserts the object set after import).
  Unit recipe: `xstep.cascade.unit=MM` with `write.step.unit=UM` (numbers
  unchanged, labelled µm), after `STEPControl_Controller.Init_s()` and
  `STEPCAFControl_Controller.Init_s()`. **Bands and all partitions are written
  pre-split in the solids** (the SGB imprint), so no AEDT Imprint is needed;
  face identity = per-solid OCC face order (observed equal to AEDT face-id
  order on 4 cases × 2 solids; not documented, so every import is guarded by
  face counts plus per-face area/centre checks, fail closed). AEDT does not
  expose face names to scripting. Shared faces are duplicated per solid in
  AEDT (normal AEDT practice). Free faces (lumped-port sheets) cannot travel
  in STEP: PyAEDT recreates them natively from sidecar geometry. Materials
  and Perfect E boundaries were assigned from sidecar group names with 0
  lookup failures. AEDT native Imprint works but returns no face
  bookkeeping (identity only by centroid+area matching) — fallback only.
  STEP sheet bodies at µm scale do not import reliably — not used.
- **Human decisions (in chat)**: D1 — Route C is IN v1 scope (retained
  conductor volumes as a third labelling policy). D6 — remove both old
  geometry paths when the new pipeline lands; no compatibility layer (V1 is
  not finished). D2 — Human criterion: thresholds must prevent malformed
  mesh elements; the Lead proposes concrete values from mesh-quality
  evidence on extreme-case fixtures for Human approval. D5 — pending Human
  explanation.
- **D5 decided by the Human**: surfaces are attributed **per Entity** by
  default (seams between different Entities are kept as required edges),
  so every Entity's Surface-EPR contribution is separable; reports aggregate
  by **interface type** (MA/MS/SA) and can show totals; Net is used only where
  a backend needs electrical identity (Electrostatic terminals/grounds).
  Overlap rule (Lead default, reversible): where two Entities of one Net
  overlap, the overlap region belongs to the higher declared entity
  priority; equal or missing priority fails closed naming both Entities.
- **Relation to develop's fixed-grid arrangement** (Codex, in develop since
  #55): it lives inside today's SGB, in source preparation and parts of
  planning (polygon Booleans, junction partition, domain occupancy) on a
  declared grid. In the redesign it maps to S0–S1 (source normalisation on the
  declared grid); its `GeometryGridContext` is the natural carrier of the
  model grid g. S4 onward is replaced by the new arrangement/assembly/emission.
- **Route C meaning (Human)**: conductor volumes are retained and meshed
  inside (conductor interior is a solution domain), with conductor–dielectric
  and conductor–vacuum interfaces as shared faces; purpose: finite
  conductivity today and superconducting material models in the future.
  Labelling policy only (S7); no separate geometry pipeline.
- **Per-Entity group names (Lead, follows D5)**: primary surface groups are
  `<kind>::<qualified entity id>[::band=k]` (e.g. `MA::chip/q0/pad::band=2`,
  `SHEET::chip/q0/pad::band=2` for Route A sheets); interface-type totals are
  sidecar unions (`MA`, `MS`, `SA` lists); Net-level unions exist only for
  Electrostatic terminals/grounds. Route C conductor-interior groups:
  `VOL::<qualified entity id>`. Final spelling is settled when the prototype
  implements D5/Route C.
- **D2 decided by the Human**: v1 sets NO sliver thresholds. Quantisation,
  phantom and same-label seam slivers are removed by construction; source
  near-coincidence is treated as the intended geometry and is only reported
  (feature size, location, lineage). No derived-to-source snapping and no
  build failure on narrow faces in v1. Reopen only if real layouts or the
  extreme-case fixtures show malformed mesh elements caused by narrow faces.
- **After review R3 (Lead)**: (1) curved walls whose arc carries a foreign
  vertex are split at that vertex by a vertical edge (top arc split too);
  each piece is an exact cylinder on its own realised circle; the direct
  writer is exact per piece, not "for any number of boundary edges".
  Polygon-only circles never produce cylinder walls. (2) No Z threshold:
  levels are the exact declared Z values; slabs thinner than g are kept and
  reported (consistent with D2). (3) The extreme-case catalogue must record
  mesh quality and drift per fixture so the D2/D7 reopen triggers can be
  evaluated. (4) v1 curve classes: lines and circular arcs; interpolation
  splines and B-splines fail closed until a later stage.
- **D7 (Lead recommendation, confirmed)**: v1 records drift only (per-edge
  deviation in the sliver report + structural validation). Revisit
  bounded-drift ISR if drift above 1 g appears on real layout inputs or in the
  extreme-case fixture catalogue; adversarial random segment sets (1.417 g,
  1.925 g, `arrangement_guarantees.py`) are stress evidence, not the trigger.
- **Prototype timings are not final**: scaling numbers are noisy (45–74 s for
  the same N=20000 case) and polygon-only; the production implementation
  re-measures under declared load. No prototype re-runs for docs.
- **Group naming refined (Lead, stage 1e)**: primary surface groups carry the
  face kind: `<kind>::<qualified entity id>::<face kind>[::band=k]` with face
  kind in {top, bottom, sidewall, sheet, interface}; Route A sheets
  `SHEET::<entity>::sheet[::band=k]` and listed in the MA and MS
  interface-type unions; `surface_epr_rows` mark the side of two-sided
  internal faces; the sidecar carries `mesh_hints` (band boundaries,
  refinement curves, curved edges; no sizes).
