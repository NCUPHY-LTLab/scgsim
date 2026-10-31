# Handoff: SCGSim documentation ↔ OrcaPen SC PDK alignment

Context for the agent that aligns the rewritten SCGSim documentation with
OrcaPen SC PDK (`OrPenStrike/orcapen-sc-pdk`). Delete this file when the work is
done.

## Background

The SCGSim docs (`README.md`, `docs/`) were rewritten on 2026-10-30 to follow
[DOCS_STYLE.md](DOCS_STYLE.md): reader-first, one page type per page, no
lifecycle/hash/evidence wording in reader pages, Python API as the main path,
notebooks as an alternative. Read DOCS_STYLE.md first; it is the standard for
everything below.

Everything that depends on OrcaPen SC PDK content was **deliberately left
as-is or only moved**, not verified. That is this handoff's scope. Ownership:
OrcaPen SC PDK owns component geometry, materials, stacks and the notebooks;
SCGSim owns the API and the docs. Resolve owners and routes through
`$scq-collaboration-roles` before editing in the OrPen repository.

The pre-rewrite docs are backed up outside the repo; `git diff` against
`develop` shows every change.

## What changed that touches OrPen

| Page | Change |
|---|---|
| `README.md` | No longer links the SimplePad notebook folder. Links only the tutorials. |
| `docs/course/installation.qmd` | Holds every environment. Step 3 installs the beginner pin `51985d9`; "Environments for other examples" holds Circular Pad (`ecc9062`, `#env-circular-pad`) and Xmon (`35b4dc1`, `#env-xmon`). Other pages link there in one line. |
| `docs/tutorials/simplepad-model.qmd` | New shared page: the SimplePad component, Nets and `LiteratureCentral` surface-film presets. Both tutorials start from it. |
| `docs/tutorials/capacitance/{prepare,run,analyze}.qmd` | Replaces `capacitance.qmd` (an alias redirects the old URL). Rewritten using the SCGSim Python API. The historical pre-v3 SimplePad **Q3D** result (87.0 fF) and its notebook were removed from the tutorial. The Palace notebook is a short "Use the notebook instead" section. |
| `docs/tutorials/resonance-epr/{prepare,run,analyze}.qmd` | Replaces `resonance-epr.qmd` (an alias redirects the old URL). Palace/HFSS tabs. Reference values rounded to 3–4 significant figures. Notebook constants are no longer listed in prose. The separate 1.0.2rc1 HFSS EPR derivation story was condensed. |
| `docs/course/palace-eigenmode.qmd` | The Circular Pad study was condensed into "Example: Circular Pad LC study". The detailed provenance (OrPen `c79b91e` producer export, wheel SHA-256, the later `1590199` checkpoint, the VTU sidewall-cell note) was dropped from the page. |
| `docs/course/working-model.qmd` | The Circular Pad component code moved to palace-eigenmode. |
| `docs/examples.qmd` | "More public examples" keeps the three `35b4dc1` notebook groups, with a note that they may use older SCGSim calls. |

## Tasks

1. **Check the tutorial code against the notebooks.** For SimplePad at
   `51985d9`, check that `docs/tutorials/simplepad-model.qmd` and the
   `capacitance/` and `resonance-epr/` tutorial pages match the notebooks' component call
   (`square_pad_capacitor(pad_side_um, gap_um, die_side_um,
   junction_width_um)`), the Entity IDs `chip/PAD` and `chip/GROUND`, the
   port `chip/o_junction_lumped`, the layer `D0_TOP_M1`, and the preset names
   `LiteratureCentral_{MA,MS,SA}`. The tutorial code was checked against
   SCGSim's source but has not been run.
2. **Use one PDK pin, if possible.** The docs currently use three OrPen
   revisions: `51985d9` (SimplePad), `ecc9062` (Circular Pad) and `35b4dc1`
   (Xmon, CPW). Decide whether one current revision can serve all examples.
   If it can, update every link and install command, and remove the
   per-example environments. DOCS_STYLE §8 asks for one pin per example, given
   once.
3. **Circular Pad facts.** Confirm the condensed claims in
   `palace-eigenmode.qmd#circular-pad-lc-study`: 5.573 GHz, final indicator
   0.119 against 0.02, four passes, PCG warnings, junction participation
   −0.995, 114 197 tetrahedra and 846 008 unknowns. Also decide whether the
   fact that Fresh08 ran from OrPen `c79b91e`, not the linked `ecc9062`, must
   be stated on the page or only in the OrPen evidence README.
4. **SimplePad Q3D.** The OrPen `aedt_q3d` notebook uses the removed pre-v3
   GDS-import input. Either migrate it to Q3D body-v3 (`prepare_q3d_from_geometry`),
   or drop it. If it is migrated and run, add a Q3D tab to the capacitance
   tutorial.
5. **Xmon pages.** `docs/tutorial-xmon-input.qmd`, `docs/tutorial-xmon-palace.qmd`
   and `docs/tutorial-xmon-aedt.qmd` were not rewritten, except for the added
   install block. Apply DOCS_STYLE (tutorial template, numbered steps, an
   expected result with a tolerance, a short notebook section).
6. **`docs/notebook-ux.qmd`.** This page holds rules for OrPen notebook
   authors. Rewrite it with the OrPen owner, or move it to the OrPen repository
   and link it from SCGSim.
7. **CPW and other public notebooks.** Check that the `35b4dc1` notebooks
   listed in `docs/examples.qmd#public-inputs` still run with the current API.
   This matters most for `ComponentSimulation/CpwFiniteGround`, which has a
   pre-v3 Q3D notebook. Remove or update any that don't.
8. **Reference values and tolerances.** The tutorials say "expect agreement to
   the first or second decimal place" (capacitance) and "to two or three
   significant figures" (resonance). These were estimated, not measured.
   Confirm them with a fresh run on current SCGSim and the chosen PDK
   revision, or correct them.

## Claims to verify (not OrPen-specific)

These were written in plain language during the rewrite and should be checked
by someone who knows the physics and code:

- HFSS Eigenmode/EPR uses a Perfect E outer boundary by default
  (`tutorials/resonance-epr/analyze.qmd`, `palace-eigenmode.qmd`, `eigenmode-epr.qmd`).
- A "mask margin" is defined in `eigenmode-epr.qmd` as excluding a band next
  to metal edges. Check this against the actual inset semantics.
- `save_fields` / `save` is described as the number of field solutions to
  save.
- `ElectrostaticSim.mesh()` requires `set_surface_epr()`. The capacitance
  tutorial uses the full presets, while the Palace Electrostatic guide uses
  `specs={}`. Make the two consistent, or fix the API so the call is optional.

## Not in this handoff (SCGSim-internal)

- `docs/specs/aedt-runtime.qmd` is still about 1700 lines. Its status lines,
  dev-log narratives and receipt version history were removed; the receipt
  section moved to `docs/provenance.qmd#aedt-receipt-cohorts`. It still needs
  to be split into per-family reference pages.
- `docs/architecture.qmd` and `docs/ownership.qmd` still overlap.
