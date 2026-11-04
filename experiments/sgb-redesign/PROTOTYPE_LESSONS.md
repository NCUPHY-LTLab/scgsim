# Lessons from the SGB prototype

The prototype (`sgbproto`, phases 1, 1b, 2 and 3) proved the design in
`DESIGN_BRIEF.md` before the production package `scgsim.geometry.sgb` existed.
Production has replaced it, so its code is no longer kept here. These are the
lessons it taught, each with what production does about it now.

## Geometry and arcs

- **Offset carriers are exact only up to the grid.** The offset of an oblique
  edge involves irrational positions, so curve end points are snapped like every
  vertex: exact carriers, snapped end points. Measured effect: at most 0.63 grid
  units on arcs and 1.6e-6 relative on band areas against KLayout.
- **Circles must be split.** A full circle is a loop edge and a half circle
  shares both end points with its chord, which the DCEL and the BREP topology
  cannot represent. Circles are four quarter arcs; reflex arcs split at 90°.
- **An arc witness merges back only at a clean vertex.** Chords of an arc merge
  only where no foreign carrier makes a hot pixel between them; a crossing
  splits the arc and each piece is realised through its own snapped end points.
- **The arc witness is expensive before it merges.** Hot pixels grow like
  √(r/g) per arc (an 8 µm pad went from 69 to 516 hot pixels). Budget for it in
  arc-heavy layouts.
- **Volumes are exact only for straight edges.** Each cylinder wall adds a
  floating-point circular-segment term; arc models match analytic volumes to
  about 1e-9 relative, not exactly.
- **Floating point must not decide band membership from one point.** The
  prototype classified a cell by the distance of one representative point,
  which can land on the medial axis. Production labels a face by area and fails
  closed when a face straddles two bands.

## Topology and identity

- **Wall face ids cannot include the vertical edges**, because column segments
  depend on the global set of cut levels. A wall id hashes its domains, its
  chain and its z range; vertical edges are derived data of the walls.
- **Wall coalescing needs two extra cut rules**: a sheet face along a wall chain
  cuts the wall even without a vertex on it, and a wall column splits where a
  neighbouring wall run ends or a horizontal face has a vertex on the column.
  Without them neighbouring faces do not share whole edges.
- **Contact between different Nets is an error in every route**, including a
  contact that exists only after snapping to a coarser grid. Same-Net contact is
  normal, except that finite metal touching only at a corner pinches the volume
  (a vertical edge used four times) and fails validation.
- **Refinement curves must never vanish silently.** A refinement piece in a cell
  with no face, or in a face of no emitted volume, is an error.

## Emission

- **Building solids through the Gmsh API cannot guarantee their sense.**
  `addSurfaceLoop` can produce inside-out solids on large layouts, and
  `addSurfaceFilling` is not exact on curved walls. Production writes the BREP
  directly with every orientation explicit (decision D8).
- **`addPhysicalGroup` cost grows with the number of groups**, so the Gmsh-API
  path scales badly; the direct writer does not pay this.
- **Arc centres need construction points in Gmsh** that must be removed again,
  or they mesh as stray nodes and break one planned id per native entity.

## Performance

- **Noding rounds are a barrier.** Iterated snap rounding re-routes chords across
  tiles, so each round needs the union of hot pixels; noding is a per-round
  fork/join, not one job per tile.
- **Face extraction is not tiled.** Faces span tiles (a ground plane is one face
  with every cutout as a hole), so the DCEL and dissolution stay serial.
- **CPython caps the parallel gain.** Forked workers inflate compute through
  copy-on-write refcount faults, and a serial fraction near 0.45 caps the gain.
  Garbage-collection pauses were about 30 % of the time. The Human accepted the
  serial performance as sufficient; parallel speed-up is not a goal.

## Decisions that replaced prototype behaviour

- Groups are named by entity id; the prototype's `SHEET::<net>` and `VOL::`
  names are gone, and a sheet's Surface-EPR rows list `face_sides`.
- A lumped port lies on its hosts' authored support plane, not in the metal
  mid-plane as an embedded face.
- Inset band seams stop at a port's boundary, so a port is one face; the
  prototype cut a port into seven faces.
- Inset bands are measured from gap edges only, not from the whole support
  boundary.
