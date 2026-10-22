## Change

- Behavior or maintenance scope:
- Semantic state: `CONVERGING` / `ACCEPTED` / `STABILIZED`:
- Delivery state:

## Analytic correctness evidence

- Independent public analytic regression (normalization, participation, and group identities):
- If no regression case is included, why not:
- Alternative evidence used:
- Remaining gap:

A historical aggregate whose calculation procedure is unknown is not a correctness
oracle; do not derive a tolerance from it.

## Returned-data compatibility

- Status: `COMPLETED` / `IN PROGRESS` / `NOT RUN` / `NOT APPLICABLE`
- Permitted artifact hashes, identity, and receipt references:
- Raw-record preservation in the public result, including unknown values:
- Missing-input, content-mismatch, and identity-mismatch outcomes:

Keep private values, names, paths, and payloads out of public text.

## Public CI and Tier A evidence

Public CI uses repository-visible inputs. Pull request events, including
drafts, pushes to `develop` or `main`, and manual dispatch run a Python
3.12/3.13 matrix that checks the package-version contract, builds the
distribution, and verifies editable-source and wheel imports. The
Python 3.12 offline job runs the full public non-native test suite, strict
Pyright on the three configured modules, and Ruff F on tracked Python files.
A separate Python 3.12 `docs-smoke` job runs current source examples through
file-only preparation with the pinned public PDK; the required job passes only
when all three jobs succeed. CI does not invoke AEDT or Palace native solvers
or read private golden data. Pages rendering on `develop` and `main` pushes or
manual dispatch is separate from pull-request CI.

- Affected Tier A behavior: `yes` / `no`
- Assigned private golden evidence: `COMPLETED` / `NOT RUN` / `NOT APPLICABLE`
- If `NOT RUN`, state the unavailable data or access and the affected claim:

Keep private golden data outside this repository. A public CI pass does not
substitute for required Tier A evidence.

## Delivery

- Target branch:
- Required PR checks: `pending` / `complete` (identify the actual checks):
- Develop required-check/no-bypass protection: configured; candidate's required hosted check: `pending` / `complete` (link the actual run):

CI status does not determine semantic acceptance. See the
[engineering convergence status](../docs/convergence-status.qmd).
