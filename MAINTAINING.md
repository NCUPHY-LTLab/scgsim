# Maintaining SCGSim

This is the maintainer guide: what each CI job checks, how the documentation
site is built and published, and how the generated pages are refreshed. The
contributor pages, in reading order, and the commands that check a change
before a pull request are on
[Contribute to SCGSim](docs/contribute/index.qmd).<a id="implementation-and-project-references"></a>

- [Sources and generated output](#sources-and-generated-output)
- [Source ownership](#source-ownership)
- [Writing documentation](#writing-documentation)
- [Native presentation](#native-presentation)
- [Documentation builds](#documentation-builds)
- [Bilingual authoring](#bilingual-authoring)
- [Generated API references](#generated-api-references) and the [generated code map](#generated-code-map)
- [CI jobs](#ci-jobs)
- [Fix pull requests](#fix-pull-requests)

## Sources and generated output

`src/scgsim/` owns reusable product code. `docs/` and the READMEs own authored
explanations; `_quarto.yml` and `zh-TW/_quarto.yml` own site navigation and
execution settings. `scripts/` owns build and version tools; `tests/` owns the
maintained package tests. Generated HTML, search indexes and the packaged
knowledge bundle are derived output. Public component sources and executable
notebooks belong to OrcaPen SC PDK.

## Source ownership

Palace and AEDT backend implementations, their tests and their technical
documentation are separate source responsibilities, as
[AGENTS.md](AGENTS.md) describes. Native solver observations and result
interpretation stay within their stated technical boundaries. Coordination
and delivery follow [AGENTS.md](AGENTS.md) and the shared `$github-flow`;
a major architecture, cross-package contract or system-wide proposal goes
through that same process.<a id="larger-proposals-across-projects"></a>

## Writing documentation

Follow [DOCS_STYLE.md](DOCS_STYLE.md) for SCGSim's sources, packaging fields,
writing rules and terminology. The shared
[docs-design Skill](https://github.com/arfiligol/agent-skills/blob/main/docs-design/SKILL.md)
owns reader principles; the
[quarto-docs Skill](https://github.com/arfiligol/agent-skills/blob/main/quarto-docs/SKILL.md)
and its Askr adoption reference own site implementation and theme consumption.

## Native presentation

The site uses the complete upstream Askr extension without local CSS or JS.
The Askr revision and the Quarto version of the Pages build are the constants
`ASKR_COMMIT` and `QUARTO_VERSION` of `scripts/build_docs_site.py`; the
authoring copy in `_extensions/` is the same Askr release and states the
Quarto range it supports. Use native Quarto structures for tables, callouts
and navigation; keep source identity separate from generated presentation
identity.

## Documentation builds

`scripts/pages_versions.json` is the version catalogue. Its
`retired_development_lines` names release lines whose development root is no
longer published: `v<line>-dev/` then leads to the current development site,
so links made by installed prereleases of that line still reach the page. Add
a line there when the development version moves to the next release line.
`scripts/build_docs_site.py` freezes source revisions and metadata, renders
clean snapshots with code execution disabled and assembles the combined site.
Current `main` and `develop` use their own content; retained historical
versions keep their original README and Quarto bodies with a documented
presentation overlay.

The root selects the latest stable version. Changing version keeps the current
page where it exists, otherwise opens that version's home. `/main/` and
`/develop/` are aliases. The inline catalogue is the same on every version
site: the publisher writes the menu (`askr-versions.json`) of each language
project of every version, before rendering and beside the rendered pages, and
each root in it is relative to that project's folder, as Askr reads it. The
committed `askr-versions.json` files serve a local preview only. Package
version, content revision, presentation revision, Askr revision and branch
are separate fields in the generated build records.

The Documentation Pages workflow builds on pushes to `develop` and `main` and
on manual dispatch; pull requests do not publish. A failed required render
prevents publication and leaves the last deployed site in place. Source
changes reach `main` by promotion from `develop`; stable package metadata
belongs to `main`. Check a version or catalogue change with
`tests/test_docs_site_build.py` and `tests/test_documentation_links.py`.

## Bilingual authoring

The root Quarto project owns the English site; `zh-TW/` owns the Traditional
Chinese site, with the same page paths. Each project has its own navigation
and search index, and Askr's language selector links the two homes. Keep page
anchors, code, units and scientific meaning aligned when you translate.

`quarto render --no-execute` at the repository root renders the English site.
Its post-render step, `scripts/build_bilingual_docs.py`, copies the shared
Askr extension, `docs/assets/` and `docs/images/` into `zh-TW/`, renders the
Chinese project into `_site/zh-TW/` (with the `pages` profile when
`zh-TW/_quarto-pages.yml` exists) and writes the redirects of pages whose
sections moved (`SPLIT_PAGES`). The staged copies are derived, not a second
editable theme. No notebook cell and no solver runs. The root project's render
list excludes `zh-TW/**`. Serve the combined site statically so that each
language keeps its own navigation and search, and open `/` and `/zh-TW/`:

```bash
python3 -m http.server 4321 --bind 127.0.0.1 --directory _site
```

## Generated API references

Edit the authored usage Markdown under `docs/reference/api/` and
`zh-TW/docs/reference/api/`, or the owning public Python declarations, then
refresh and check the derived references:

```bash
python scripts/generate_knowledge_reference.py
python scripts/generate_knowledge_reference.py --check
```

The committed QMD pages are derived output. The fast tests check that they
are fresh; package builds copy them into the offline knowledge bundle without
importing SCGSim or starting a solver.

<a id="generated-pages"></a>

## Generated code map

The stage tables of the [code map](docs/contribute/code-map.qmd)
(`docs/contribute/_code-map-stages.md` and its Chinese copy) are generated from
the docstrings of the package folders. To change a row, edit the folder's
`__init__.py` docstring, translate the changed text in
`scripts/code_map_zh_tw.toml`, and run:

```bash
python scripts/generate_code_map.py
python scripts/generate_code_map.py --check
```

The order of the rows is stated once, in `AREAS` of that script, and each
entry of the translation file records the English it translates. The script
refuses a package folder `AREAS` does not list, a stage whose docstring lacks
one of the lines `In:`, `Out:` and `Next:`, a folder without a translation
entry, an entry for a folder that is gone, and an entry whose recorded English
is no longer the docstring's text; that refusal prints the new English to
record and translate. `tests/test_code_map.py` fails while the tables differ
from the docstrings and their translations.

<a id="engineering-convergence"></a>

## CI jobs

A change reaches `develop` through a pull request with the required checks
and no bypass. Each CI job is its own check, so a failure names the job:
`change-scope`, `build-and-import (3.12)`, `build-and-import (3.13)`,
`fast-tests`, `quality`, `docs-smoke` and `pdk-tests`
(`.github/workflows/ci.yml`). A job skipped by its condition (a Draft pull
request, or the prose-only lane) reports success. A failed classification also
skips the other jobs, so `change-scope` is itself a required check. Failed or
cancelled jobs are failures.

`change-scope` classifies the exact event comparison first. A Draft pull
request runs only this classification; marking it Ready starts the other
jobs. A change of Markdown prose only, in `README.md` or `docs/**/*.md`, gets
a local check of its Markdown link paths instead of the product jobs.
Documents with inline-code ambiguity, executable or unsupported content, QMD
and notebook files, shared settings, dependencies and source changes take the
product jobs, and so does a comparison that cannot be read.

The product jobs run on a non-Draft pull request, on pushes to `develop` and
`main` and on manual dispatch:

| Job | What it checks |
|---|---|
| `build-and-import (3.12)`, `(3.13)` | The package version, the distribution build, the editable import and an isolated wheel import, on Python 3.12 and 3.13. |
| `fast-tests` | The public offline suite except the `slow`, `pdk` and `private_golden` tests. |
| `quality` | `scripts/check_quality.py`: Ruff formatting and lint, the readable-style check, Pyright (standard mode, with the strict files listed in `pyproject.toml`) and the import-layer contracts of `.importlinter`. |
| `docs-smoke` | The current source examples, through file-only preparation, with the pinned public PDK (`scripts/check_docs_examples.py`); it installs OCP at the version of the `aedt` extra, because the AEDT examples export a STEP. |
| `pdk-tests` | The `pdk`-marked tests against the public OrcaPen SC PDK at the commit `README.md` installs, from that PDK's frozen lock with the current SCGSim wheel. |

`fast-tests` and `quality` install the `dev` group, which includes `occt`, so
Pyright resolves OCP and the `occt`-marked checks run. The test jobs pass
`--fail-on-skip`: they install every dependency their tests need, so a skip
fails the run instead of passing unnoticed.

The Nightly workflow (`.github/workflows/nightly.yml`) runs at 18:30 UTC on
`develop`, and on any branch by manual dispatch:

| Job | What it checks |
|---|---|
| `full-tests` | The whole public suite, with the `slow` tests. |
| `docs-render` | The Pages render of `scripts/build_docs_site.py`, without deploying it. |
| `roster` | Each layout of `scripts/roster_cases.py`, built with the PDK at the `pdk-tests` pin and with Surface-EPR bands, must pass `simulation_drc` and then mesh within the default time budget, in a vacuum region padded 0 µm in x and y and 1000 µm below and above (`scripts/check_roster.py`). It uploads one record per layout as the `roster-records` artifact. |

CI uses public inputs only and never runs AEDT or Palace. The Documentation
Pages render is separate from CI: it runs on pushes to `develop` and `main`,
not on pull requests.

## Fix pull requests

A fix pull request includes a regression case, or says why it has none, what
evidence it uses instead and what gap remains. For a calculation change, use
independent public analytic cases to establish normalization, participation
and group identities, and report them apart from compatibility with returned
data. A historical aggregate whose calculation procedure is unknown is not a
correctness oracle: keep it as history and derive no tolerance from it.

The `private_golden` tests read returned data held outside the repository.
They run only with an explicit flag and a manifest, and public CI excludes
them; the manifest is an input to the reader, not permission to access or
publish its data:

```sh
python -m pytest tests/test_private_golden.py \
  --run-private-golden \
  --private-golden-manifest /path/to/authorized-manifest.json
```

A report of such a run names only permitted artifact hashes, identities and
receipt references, never private values, names, paths or payloads. Use the
[pull request template](.github/PULL_REQUEST_TEMPLATE.md) and the linked Issue
for the scope and open items of the change.
