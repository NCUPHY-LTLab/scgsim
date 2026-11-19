# SCGSim Documentation Guide

Use this guide to locate the owning source, preserve package metadata and
choose the correct terminology when editing SCGSim documentation. Shared
[docs-design](https://github.com/arfiligol/agent-skills/blob/main/docs-design/SKILL.md)
owns reader principles; shared
[quarto-docs](https://github.com/arfiligol/agent-skills/blob/main/quarto-docs/SKILL.md)
owns Quarto implementation and the Askr adoption policy.

## 1. Readers

The user pages (Get started, Guides, Concepts and Reference) serve chip
designers and physicists who know superconducting circuits and Python and are
learning SCGSim, not its internals; write them in plain words, not in a
technical way. Contributor material belongs in the Contribute area
(`docs/contribute/`), whose entry page gives its reading order, and in
[MAINTAINING.md](MAINTAINING.md). Contribute pages may be technical; they must
still be correct and current.

### How to write a page

- Open with what the reader gets: the answer, the result, or what they will
  do. Then the context.
- Define every term a user may not know in plain words where it is first used
  (Net, Entity, Route, region, MA/MS/SA, snapshot, handoff, sidecar), or move
  the material to Contribute.
- Give an abstract idea a dimensioned example or a figure, with real numbers.
- Say a limitation once, where the reader meets it, and link to it elsewhere.
- Show only real results: from a run made for the page, from a run folder or
  figure in the repository, or a recorded result named as such. Never round a
  number into a new one.
- Leave out process noise: no commit hashes (a pinned source link may carry a
  version), no issue numbers as the provenance of a change, no dates of checks,
  no internal run names, no lifecycle states. One link to an Issue where a
  user can follow a known limitation is fine.
- Keep headings short and plain: no code or formula as a heading, and no long
  phrase repeated in every heading.
- Write no prose that only restates a table or a code block, and do not repeat
  a passage across pages.
- Write a Chinese page as natural Traditional Chinese (Taiwan), not a
  sentence-by-sentence translation: short sentences, Chinese punctuation, the
  terms of [section 5](#chinese-terms) and 你 for the reader, with identifiers,
  API names and units unchanged.

## 2. Owning sources

| Content | Owning source |
|---|---|
| Homepage | Each language's README, included by its `index.qmd` |
| Installation | [Install SCGSim](docs/start/install.qmd) |
| Solver execution and host prerequisites | [Run a prepared model](docs/guides/run-elsewhere.qmd) |
| Returned-result loading and interpretation | [Convergence and reports](docs/guides/convergence-and-reports.qmd) |
| Contributor reading order and checks | [Contribute to SCGSim](docs/contribute/index.qmd) |
| Code map stage tables | The package folders' docstrings; regenerate through `scripts/generate_code_map.py` ([MAINTAINING.md](MAINTAINING.md#generated-code-map)) |
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
status labels. Preserve existing resource IDs, also when a page moves, and keep
`docs/examples` (the page `docs/start/choose-a-simulation.qmd`) as the sole
resource whose kind is `start`. A moved page keeps its old URL in `aliases:`
and its old section anchors (an empty `<span id>` where a section's content
now is); a page whose sections go to other pages gets a `SPLIT_PAGES` entry in
[scripts/build_bilingual_docs.py](scripts/build_bilingual_docs.py) instead of
an alias.

Keep model prerequisites, physical assumptions, units and result meaning with
their owning documentation. Retain source versions in example links and
execution details in the corresponding study or contributor records.
Model-specific studies belong in Research (`docs/contribute/research/`);
reusable API behavior and implementation checks belong with the tool
contracts in Reference.

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

### Chinese terms

Chinese reader pages use these Taiwanese terms and address the reader as 你.
API names, interface strings and the solvers' own names (HFSS `Eigenmode`,
`Driven Terminal`, Q3D `Signal`) stay as they are.

| English term | Taiwanese term | Note |
|---|---|---|
| port | 埠 | Not 端口 or 連接埠. A lumped-port sheet is 集總埠片. |
| terminal | 端子 | The conductor a capacitance solve sets to 1 V. Not 終端 (終端機 is the terminal window). |
| eigenmode | 本徵模態 | Not 本徵模. |
| resonant mode | 共振模態 | |
| inset margin | 邊距 | The distance from a gap edge to a band boundary. Not 內縮邊距 or margin. |
| band | 條帶 | One inset band; a band family is 條帶族. |
| snapshot | 快照 | The immutable result of `plan.prepare()`. |
| coupon | 試片 | The small structure around the device that is simulated. Not 試樣 or coupon. |
| region | 區域 | For a region of the model, not 區塊. |
| mesh | 網格 | Not 格網. |
| vacuum region | 真空區域 | The box of empty space the solver includes. Not 真空區, 真空範圍 or 真空盒. |
| Net | Net | Kept in English. On first use, explain it: 一塊具名、作為同一個導體的金屬. |
| you | 你 | Not 您. |

## 6. Bilingual and derived output

Keep the two languages' page paths, anchors, code, units and scientific meaning
aligned. The SGB design pages (`zh-TW/docs/contribute/sgb-design/`) exist in
Chinese only. Use the native Askr components and site options described by the
shared adoption policy. The root extension is shared with the Chinese build;
staged extensions, rendered sites and packaged knowledge remain derived output.

Validate the affected authored source and links. Render affected reader pages
when their presentation changes; use the existing delivery checks when a
shared site dependency changes. Documentation validation does not itself
authorize a Solver run or publication.
