# Maintaining SCGSim

This repository reference covers documentation builds and package maintenance.
The secondary [maintainer guide](docs/ownership.qmd) links the implementation
and project references. Researchers can start with the
[course](docs/examples.qmd) without a parent workspace.

## Sources and generated output

`src/scgsim/` owns reusable product code. `docs/` and README own authored
explanations; `_quarto.yml` owns site navigation and execution settings.
`scripts/` owns build/version tools; `tests/` owns maintained package tests.
Generated HTML, search indexes and packaged knowledge are derived output.
Public component and executable notebook sources belong to OrPen SC PDK.

## Source ownership

The Architecture, Palace and AEDT Leads own the source responsibilities listed
in [AGENTS.md](AGENTS.md). Architecture decides integration order; SCQ Integration
performs public privacy review and protected merges. Resolve current Lead
identities and shared delivery rules through `$scq-collaboration-roles`.

Existing writers retain their assigned paths and running services. Coordinate
with the assigned writer before changing a shared path or taking over a scope.

## Writing documentation

Follow [DOCS_STYLE.md](DOCS_STYLE.md) for SCGSim's source, packaging and
terminology requirements. The shared
[docs-design Skill](https://github.com/arfiligol/agent-skills/blob/main/docs-design/SKILL.md)
owns reader principles; the
[quarto-docs Skill](https://github.com/arfiligol/agent-skills/blob/main/quarto-docs/SKILL.md)
and its Askr adoption reference own site implementation and theme consumption.

## Native presentation

The site uses the complete upstream Askr extension without local CSS or JS.
The authoring extension and versioned site builder use Askr v0.13.3 at
`243d3dbdd234004bbe0538b9518fb76027903915`, supporting Quarto >=1.9.38 and <1.11.
The local documentation checks use Quarto 1.10.19; the Pages workflow selects
1.10.18 within that supported range.
Use native Quarto structures for tables, callouts and navigation; keep source
identity separate from generated presentation identity.

## Documentation builds

`scripts/pages_versions.json` is the version catalogue.
`scripts/build_docs_site.py` freezes source revisions and metadata, renders
clean snapshots with code execution disabled and assembles the combined site.
Current main/develop use their own content; retained historical versions keep
original README/Quarto bodies with a documented presentation overlay.

The root selects latest stable. Version changes retain the current page where
it exists, otherwise use that version's home. `/main/` and `/develop/` are aliases.
The inline catalogue is consistent across version sites. Package version,
content revision, presentation revision, Askr revision and branch remain
separate fields in generated build records.

The Documentation Pages workflow builds on develop/main pushes or manual
workflow dispatch; pull requests do not publish. A required render failure
prevents publication and leaves the last deployed site in place. Source changes
follow develop-to-main promotion; stable package metadata belongs to main.
Version/catalogue changes should be checked with the existing maintained
version tooling and the delivery packet's assigned observations.

## Generated API references

Edit the authored usage Markdown under `docs/api/` and `zh-TW/docs/api/`, or the
owning public Python declarations. Refresh and check the derived references:

```bash
python scripts/generate_knowledge_reference.py
python scripts/generate_knowledge_reference.py --check
```

The committed QMD pages are derived output. Ordinary tests check freshness;
package builds copy them into the offline knowledge bundle without importing
SCGSim or starting a solver.

## Implementation and project references

- [Architecture and data flow](docs/architecture.qmd)
- [Runtime and result provenance](docs/provenance.qmd)
- [Project scope and derivation](docs/goals-and-upstream.qmd)
- [Project direction](docs/roadmap.qmd)
- [Historical diagnostic archive](docs/errors.qmd)

The integrated derivation inputs were gsim downstream
`0.2.0+scq.1`, revision `8f5dc6c05255d003a9c6d8959537bcf8068379d3`,
and historical SGB revision `e74a343154c6b19b6ba32d6fb297e700cfe08ff2`.
They explain origin, not runtime dependencies, support expansion or API
compatibility. Current source geometry uses the in-tree `scgsim.geometry` API
and `scgsim.geometry.compiler` for route planning and derived construction.

## Engineering convergence

The current engineering-convergence work remains `CONVERGING`. Track semantic
state and repository delivery separately in
[the convergence status page](docs/convergence-status.qmd).

The required delivery route is a pull request with required checks and no
bypass. The literal `required` check always reaches a terminal result and
accepts only the results declared for the selected change scope. Failed or
cancelled jobs and unexpected skips remain failures.

CI first classifies the exact event comparison. Draft pull requests run this
classification and intentionally skip product jobs; marking a pull request
Ready starts the applicable checks. Proven Markdown prose changes in
`README.md` or `docs/**/*.md` receive local Markdown link-path validation.
Documents with inline-code ambiguity stay on the product path. Executable or
unsupported content, QMD/Notebook files, shared settings, dependencies and source
changes also use the product checks. An unavailable comparison also uses
the product checks.

Non-Draft product changes and manual dispatch retain the Python 3.12/3.13
package-version, distribution-build, editable-import and isolated-wheel-import
checks. The Python 3.12 offline job runs the full public offline test suite,
the unified `scripts/check_quality.py` command: Ruff formatting and lint,
readable style, standard Pyright with the three retained strict scopes, and
six import-layer contracts. That job also installs the `occt` dependency group, so Pyright
resolves OCP and the `occt`-marked OCCT checks run instead of skipping. The separate Python 3.12 `docs-smoke` job runs current source examples
through file-only preparation with the pinned public PDK. CI uses public
inputs and excludes private golden data; it does not invoke AEDT or Palace
solvers.

Pages renders the frozen version catalogue, stable pages and history on
`develop` and `main` pushes or manual dispatch, with code execution disabled.
It remains separate because this coverage differs from file-only preparation;
it is not a pull-request check.

For each fix pull request, include a regression case or explain why one is not
included, the alternative evidence used, and any remaining gap. For calculation
changes, use independent public analytic regressions to establish normalization,
participation, and group identities, and report them separately from returned-data
compatibility. A historical aggregate whose calculation procedure is unknown is
not a correctness oracle; preserve it as history and do not derive a tolerance
from it.

For an authorized real-data compatibility check, report only permitted artifact
hashes, identity, and receipt references. State whether raw records, including
unknown values, are preserved through the public result, and report missing-input,
content-mismatch, and identity-mismatch negatives. Keep private values, names,
paths, and payloads out of public text. If the check has not finished, mark it
`IN PROGRESS`.

For an affected Tier A change, report the assigned private golden status. If
required data or access are unavailable, record `NOT RUN` and leave the
associated claim open. An owner-authorized run uses the public test entry point
with an explicit run flag and manifest:

```sh
python -m pytest tests/test_private_golden.py \
  --run-private-golden \
  --private-golden-manifest /path/to/authorized-manifest.json
```

The manifest is an input to the reader, not permission to access or publish
its data. Missing required manifest input is a usage error; a skipped test is
`NOT RUN`, not a golden result. Public CI excludes the `private_golden` marker.
Keep golden data outside the public repository. Tier B remains supported and
is not reclassified by this convergence work. Use the
[pull request template](.github/PULL_REQUEST_TEMPLATE.md) and the
[convergence status page](docs/convergence-status.qmd) for current scope and
open items.

## Larger proposals across projects

Send a major architecture, cross-package contract, or system-wide proposal to
the nearest owning Lead. Before writing an execution plan, that Lead checks the
repository, branch and base revision, related source, open work, contracts, and
existing capabilities. Keep the plan in the current work packet. For authorized
delivery, the repository's Git owner completes the assigned branch and push
steps and reads back the remote result; a proposal does not transfer that
ownership.

## Historical structural acceptance

The v2 module and API structure was accepted on 2026-10-09 and later integrated
through PR34. That historical milestone describes the structure only; it does
not set the semantic or delivery state of the separate engineering-convergence
work.
