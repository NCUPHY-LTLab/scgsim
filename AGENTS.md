# SCGSim Agent Guide

`NCUPHY-LTLab/scgsim` is SCQ_Design's reusable layout, solver, runtime,
and result-production authority. Three source-owning Leads implement and
maintain their assigned parts of this repository:

| Lead | Source responsibility |
|---|---|
| SCGSim Architecture | Package direction, shared geometry and semantics, shared presentation and visualization, and common configuration |
| SCGSim Palace | Palace backend implementation and its related tests and documentation |
| SCGSim AEDT | AEDT backend implementation and its related tests and documentation |

Architecture also owns the tests and documentation for shared code. Existing
documentation-site and Preview writers retain their scopes during handoff.

Use the `develop-main` delivery profile and `$scq-collaboration-roles` for
current Lead identities, ownership, scientific consultation and delivery; it
composes the lifecycle and model-routing policies. Each Lead commits and pushes
its own topic branch. Architecture decides integration order; SCQ Integration
retains public privacy review and protected merge authority.

Each source-owning Lead may make small, tightly coupled edits directly. Delegate independent
work to the existing team when useful, including a documentation writer for
substantial writing. Assign one writer per path or exact hunk and preserve
other writers' WIP and running services. Routine documentation needs no
separate Lead approval; unresolved product or scientific meaning follows the
shared routing rules.

Read the task-relevant entry and smallest applicable contract. `README.md`,
`docs/goals-and-upstream.qmd`, `docs/architecture.qmd`, `docs/geometry-sgb.qmd`
and `docs/ownership.qmd` are reference entries, not a mandatory reading batch.
Authorized implementation includes directly related documentation, tests and
local checks under `$v1-lifecycle`; research and native execution follow the
assigned task scope.

Follow `CODE_STYLE.md` for Python changes. Run `python scripts/check_quality.py`
in the locked quality environment for the complete maintained Python scope;
readability and responsibility boundaries also require review.

External gsim and historical SGB are derivation provenance only: new work uses
in-tree `scgsim.geometry` and advanced `scgsim.geometry.compiler` APIs, never
an external consumer or fallback. OrcaPen SC PDK owns public component-simulation
notebooks; SCGSim owns no duplicate notebook source.
