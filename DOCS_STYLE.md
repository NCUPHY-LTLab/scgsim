# SCGSim Documentation Guide

Use this guide to locate the owning source, preserve package metadata and
choose the correct terminology when editing SCGSim documentation. Shared
[docs-design](https://github.com/arfiligol/agent-skills/blob/main/docs-design/SKILL.md)
owns reader principles; shared
[quarto-docs](https://github.com/arfiligol/agent-skills/blob/main/quarto-docs/SKILL.md)
owns Quarto implementation and the Askr adoption policy.

## 1. Readers

Reader documentation serves researchers who know superconducting circuits and
Python and are learning SCGSim. Contributor material belongs in
[MAINTAINING.md](MAINTAINING.md) and its linked references.

## 2. Owning sources

| Content | Owning source |
|---|---|
| Homepage | Each language's README, included by its `index.qmd` |
| Installation | [Install SCGSim](docs/start/install.qmd) |
| Solver execution and host prerequisites | [Run a prepared model](docs/guides/run-elsewhere.qmd) |
| Returned-result loading and interpretation | [Read returned results](docs/guides/convergence-and-reports.qmd) |
| Public component models and executable notebooks | OrcaPen SC PDK; link its source rather than copying it |
| API index and signatures | Public Python declarations and authored usage Markdown; regenerate through `scripts/generate_knowledge_reference.py` |
| Navigation and rendering | Root and `zh-TW/_quarto.yml`; `scripts/build_docs_site.py` owns versioned publication |

Preparation, execution and analysis may need different instructions because
they use different inputs and environments. Choose page boundaries for the
workflow; an example does not require a fixed number of pages.

## 3. Packaging and content requirements

Keep the `scgsim-kind`, `scgsim-status` and `scgsim-id` frontmatter fields on
packaged QMD pages. [build_backend.py](build_backend.py) reads them for the
offline knowledge bundle; they describe packaging and are not reader-facing
status labels. Preserve existing resource IDs and keep `docs/examples` as
the sole resource whose kind is `start`.

Keep model prerequisites, physical assumptions, units and result meaning with
their owning documentation. Retain source versions in example links and
execution details in the corresponding study or contributor records.
Model-specific results belong in Research; reusable API behavior and
implementation checks belong with the tool contracts.

Each example identifies its compatible component source and SCGSim API.
Example-specific source pairing belongs with that example; the installation
page teaches installation. SCGSim documents its own API and links to the PDK's
notebook source.

## 4. Scientific explanations and examples

Check function and argument names against their current owning declarations.
Explain prerequisites and non-obvious arguments, including units. Keep
conceptual code distinct from executable examples.

Use reported or independently established reference values and criteria.
Document the applicable model and settings; writing guidance does not create
a numerical tolerance, convergence rule or research acceptance condition.
Source and scientific owners retain the responsibilities in
[AGENTS.md](AGENTS.md).

## 5. Terminology owners

The following pages own the definitions previously collected in this guide.
Use their terminology and link to them when introducing a technical term.

| Term | Owning explanation |
|---|---|
| Entity | [Source identity, Occurrence and Net](docs/concepts/geometry.qmd#entity-occurrence-and-net) |
| Net | [Terms and units](docs/reference/terms-and-units.qmd#term-net) and [geometry contact rules](docs/reference/specs/geometry-plan.qmd#component-to-backend-geometry-contract) |
| GeometryPlan | [Terms and units](docs/reference/terms-and-units.qmd#term-geometry-plan) |
| Snapshot | [Terms and units](docs/reference/terms-and-units.qmd#term-snapshot) |
| Handoff / run package | [Terms and units](docs/reference/terms-and-units.qmd#term-handoff) and [execution](docs/guides/run-elsewhere.qmd) |
| `handoff_id` | [Preparation-side identity and result resolution](docs/guides/convergence-and-reports.qmd#palace-bind-the-independently-retained-preparation-id) |
| Resolve | [Returned-result loading and checks](docs/guides/convergence-and-reports.qmd) |
| Route A / Route B | [Geometric approximations](docs/concepts/model-and-numerics.qmd#choose-the-geometric-approximation) |

The English [Terms and units](docs/reference/terms-and-units.qmd) and
[Chinese terms and units](zh-TW/docs/reference/terms-and-units.qmd) collect reader terminology.
Chinese reader pages retain the English technical terms requested for this
project, with a Chinese explanation on first use. Keep API names and actual
interface strings unchanged.

## 6. Bilingual and derived output

Keep the two languages' page paths, anchors, code, units and scientific meaning
aligned. Use the native Askr components and site options described by the
shared adoption policy. The root extension is shared with the Chinese build;
staged extensions, rendered sites and packaged knowledge remain derived output.

Validate the affected authored source and links. Render affected reader pages
when their presentation changes; use the existing delivery checks when a
shared site dependency changes. Documentation validation does not itself
authorize a Solver run or publication.
