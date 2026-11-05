# SCGSim Agent Guide

`NCUPHY-LTLab/scgsim` is SCQ_Design's reusable layout, solver, runtime,
and result-production authority. Three source-owning Leads implement and
maintain their assigned parts of this repository:

| Lead | Source responsibility |
|---|---|
| SCGSim Architecture | Package direction, shared geometry and semantics, shared presentation and visualization, and common configuration |
| SCGSim Palace | Palace backend implementation and its related tests and documentation |
| SCGSim AEDT | AEDT backend implementation and its related tests and documentation |

Architecture also owns shared tests, package-wide documentation, the
documentation site and Preview operations. Preserve active writers and services
when coordinating changes across these scopes.

Use the `develop-main` delivery profile and `$scq-collaboration-roles` for
current Lead identities, ownership, scientific consultation and delivery; it
composes the lifecycle and model-routing policies. Each Lead commits and pushes
its own topic branch, with Git operations serialized in each checkout.
Architecture decides integration order; SCQ Integration retains public privacy
review and protected merge authority.

Each source-owning Lead may make small, tightly coupled edits directly. Delegate
independent work to the existing team when useful, including a documentation writer for
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

Use `$docs-design` for human-facing documents and `$quarto-docs` for Quarto
and Askr implementation. Read [DOCS_STYLE.md](DOCS_STYLE.md) for SCGSim's
source ownership, packaging metadata and terminology references. Shared Skills
own the reader principles and Askr consumer policy.

Follow `CODE_STYLE.md` for Python changes. Run `python scripts/check_quality.py`
in the locked quality environment for the complete maintained Python scope;
readability and responsibility boundaries also require review.

External gsim and historical SGB are derivation provenance only: new work uses
in-tree `scgsim.geometry` and advanced `scgsim.geometry.compiler` APIs, never
an external consumer or fallback. OrcaPen SC PDK owns public component-simulation
notebooks; SCGSim owns no duplicate notebook source.

## Code Review Rules

- Review against the Issue named on the PR's `Issue:` line; a PR with no
  readable Issue has unverified acceptance.
- Review in two stages. First look for counterexamples from the change's
  normal entry points, including combinations that break assumptions (two
  valid inputs paired wrongly, two calls writing one output target, a result
  that passes quality but exceeds a limit the caller set). Then weigh each
  against the Issue, the public docs and the callers you can see.
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
