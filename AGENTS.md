# SCGSim Agent Guide

`NCUPHY-LTLab/scgsim` is SCQ_Design's reusable layout, solver, runtime,
and result-production authority. It owns shared geometry, semantics,
presentation, visualization, configuration, tests, package documentation,
the documentation site, and Preview operations, plus the Palace and AEDT
backend implementations and their related tests and documentation.

Use `$github-flow` for shared ownership coordination, CI, branches, review and
delivery. Until `cross-review-gate` is verified and required here, do not enable
per-PR auto-merge; use the currently authorized merge path only after required
CI passes.

Read the task-relevant entry and smallest applicable contract. `README.md`,
`docs/goals-and-upstream.qmd`, `docs/architecture.qmd`, `docs/geometry-sgb.qmd`
and `docs/ownership.qmd` are reference entries, not a mandatory reading batch.
Authorized implementation includes directly related documentation, tests and
local checks under `$v1-lifecycle`; research and native execution follow the
assigned task scope.

Use `$docs-design` for human-facing documents and `$quarto-docs` for Quarto
and Askr implementation. Read [DOCS_STYLE.md](DOCS_STYLE.md) for SCGSim's
source ownership, packaging metadata and terminology references. Shared Skills
own the reader principles and Askr consumer policy.

Follow `CODE_STYLE.md` for Python changes. Run `python scripts/check_quality.py`
in the locked quality environment for the complete maintained Python scope;
readability and responsibility boundaries also require review.

External gsim and historical SGB are derivation provenance only: new work uses
the in-tree `scgsim.geometry` APIs, never an external consumer or fallback.
`scgsim.geometry.sgb` and `scgsim.geometry.compiler` are lower-level APIs below
the supported entry (Code Review Rules). OrcaPen SC PDK owns public
component-simulation notebooks; SCGSim owns no duplicate notebook source.

## Code Review Rules

- Review against the Issue named on the PR's `Issue:` line; a PR with no
  readable Issue has unverified acceptance.
- Review in two stages. First look for counterexamples from the change's
  normal entry points, including combinations that break assumptions (two
  valid inputs paired wrongly, two calls writing one output target, a result
  that passes quality but exceeds a limit the caller set). Then weigh each
  against the Issue, the public docs and the callers you can see.
- The supported use is a layout that enters through `GeometryPlan` and the
  simulation classes of `scgsim.palace` and `scgsim.aedt`. Inputs built by
  hand below that entry (a catalogue for `scgsim.geometry.sgb`, a
  `GeometryBuildInput`, inputs to `scgsim.geometry.compiler`) are internal or
  advanced use; their precondition is what the entry guarantees. A
  counterexample only such an input reaches is `fix` when it silently gives a
  plausible but wrong result and a cheap check at the boundary can refuse it;
  otherwise `document` or `observe`. It is not P1 by itself: for such an
  input, the P1 clause on silently wrong results applies only where a cheap
  check at the boundary could refuse it.
- Each finding states the condition that reproduces it, what the code does,
  the code or assumption that causes it, and a proposed disposition with its
  basis: `fix` (a supported use breaks a promised behavior), `document` (the
  formal flow guarantees a precondition nothing states) or `observe` (no
  known caller reaches it). "Callers outside this repository not checked" is
  a valid basis.
- Flag as P1: a supported use breaking an acceptance criterion, a public
  contract, data or security, or Human-confirmed UI; an out-of-scope input
  that silently produces a plausible but wrong result; a test double
  standing in for the result being claimed (a real solver, database,
  authorization or integration). A counterexample outside the supported use
  is not P1 by itself.
- Test doubles that isolate dependencies are fine; call counts are fine when
  the count is the contract. Do not ask for checks the Issue does not need.
- A CI change must say in the PR what runs when, before and after, the
  expected time change and what could break.
