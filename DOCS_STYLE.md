# SCGSim Documentation Style

This guide says how to write SCGSim's public documentation: `README.md`,
`index.qmd` and everything under `docs/`. Read it before you add or edit a
page. The shared SCQ rules in
[docs-source-of-truth](../../docs/reference/guardrails/docs-source-of-truth.md)
still apply. This guide turns them into concrete SCGSim practice.

The reference page is
the pad-capacitance tutorial
([Prepare](docs/tutorials/capacitance/prepare.qmd),
[Run](docs/tutorials/capacitance/run.qmd),
[Analyze](docs/tutorials/capacitance/analyze.qmd)). When in doubt, make your
page read like those.

## 1. Who the reader is

Write for a **researcher who knows superconducting circuits and Python but
has never used SCGSim**. They want to compute a quantity for their circuit
and trust it. They do not know SCGSim's internal vocabulary, its history, or
how the package is reviewed and released.

Contributors are a second, separate audience. Their material lives in
`MAINTAINING.md` and the pages listed there. Do not mix it into reader
pages.

Before writing, answer two questions in one sentence each:

- What does the reader want to get done on this page?
- What will they have, or know, when they finish?

If you cannot answer them, the page has no clear job yet.

## 2. Page types

Every reader page is exactly one of these four types. Do not mix them.

| Type | Lives in | Job | Reader's question |
|---|---|---|---|
| Tutorial | `docs/tutorials/` | Take a newcomer from nothing to one working result | "Show me it works" |
| How-to | `docs/course/` | Do one specific task with your own model | "How do I do X?" |
| Concept | `docs/concepts/` | Explain a model choice or physical quantity | "Why / what does it mean?" |
| Reference | `docs/specs/` and `docs/*.qmd` contracts | State the exact API, inputs, outputs and failure rules | "What exactly does this accept?" |

### Tutorial

- Has one path that works. Do not include alternative backends, historical
  runs or untested paths. Link those under "Next steps".
- Is split into three pages in its own folder, matching the three places the
  work happens:
  - `prepare.qmd`: build the calculation and write the run directory.
  - `run.qmd`: run the solver on that directory. It changes nothing; to
    change anything, the reader goes back to Prepare.
  - `analyze.qmd`: load the returned directory and read the result. It works
    in a new Python session and needs no solver.
- Titles the pages `<Tutorial name> 1/3: Prepare the run`, `2/3: …`, `3/3: …`.
  Prepare lists the three pages; every page ends with a `## Next` link to the
  next one.
- Each page opens with a "Before you start" box saying what it needs from the
  previous page (session names, run directory, `handoff_id`) and the time or
  resources it takes.
- Uses numbered `## Step N: <verb phrase>` sections, numbered per page. Each
  step has a short explanation, a code block the reader can copy, and what
  they should see.
- States the expected result on Prepare as an approximate value; Analyze gives
  a recorded reference value and explains which model and convergence settings to compare.
- Analyze ends with "What changes the result" and "Next steps". "Use the
  notebook instead" goes on Prepare.

### Write shared parts once

When two pages need the same setup, write it on one page and link to it.
Do not copy it.

- **Environment and installation** live only in
  [Install SCGSim](docs/course/installation.qmd), including the environments
  for individual examples. Other pages say it in one line in "Before you
  start", for example: "Set up the tutorial environment:
  [Install SCGSim](…), Steps 1–3."
- **A model used by several tutorials** gets its own page under
  `docs/tutorials/` (for example `simplepad-model.qmd`). It ends by listing the
  Python names it leaves in the session.
- **How to run a directory** (remote hosts, Slurm, AEDT) lives in
  [Run a prepared model](docs/course/execute.qmd). **How to load and judge a
  result** lives in [Read returned results](docs/course/returned-results.qmd).
  Tutorial Run and Analyze pages show only the commands for their own run and
  link there for the rest. How-to pages link there instead of repeating
  either.

### How-to

- Assumes the reader finished a tutorial. Link it as a prerequisite in the
  first sentence.
- Covers one task, for one backend. If two backends differ, write two pages
  or use tabs for only the steps that differ.
- Gives the minimal code for the task, then the options that matter.

### Concept

- Explains physics and modeling choices, not API calls. Code is rare.
- Links to the How-to or Reference page where the reader applies the idea.

### Reference

- Precise and complete, organized for lookup, not reading start to finish.
- Describes public behavior only. Private modules, internal pipelines and
  transaction internals belong in contributor docs.
- Split a reference page when it passes about 400 lines.

## 3. What does not belong in reader pages

These are the most common problems in the current docs. Remove them on
sight.

| Do not write | Where it belongs instead |
|---|---|
| Commit hashes, tree hashes, wheel SHA256, content digests | One version pin per example, in the link to the example source. Build records belong in generated output. |
| Lifecycle and review state: `ACCEPTED`, `STABILIZED`, `CONVERGING`, "Human accepted on…", "closeout", "candidate", PR numbers | Git history, release notes, internal logs |
| Evidence and audit wording: "observation", "does not establish", "makes no claim", "receipt cohort", "custody", "fresh native readback" in reader-facing prose | Contributor docs or test descriptions. Readers get a plain statement of what the result means. |
| Historical or unmigrated paths ("historical pre-v3 Q3D", "not a current execution workflow") | `docs/api-migration.qmd` or a dedicated history page, never in a tutorial |
| Internal run names (Fresh08, Route A/B/C before they are explained), private module names | Contributor docs. If a reader must choose a Route, explain it on a Concept page and link it. |
| Repeated disclaimers in every paragraph | One "Limitations" or "What changes the result" section per page |
| Long tables of notebook constants written as prose | A code block the reader runs |

The frontmatter fields `scgsim-kind`, `scgsim-status` and `scgsim-id` are
packaging metadata read by `_build_backend.py`. Keep them, but never repeat
their values in the page body.

## 4. Say what is true, plainly

Readers need to know what works, what the number means, and what they can
change. State that directly.

- **Prefer positive statements.** Say what the result is and what it depends
  on, rather than listing what it is not.
- **Put limits in one place.** A page gets at most one section on limits,
  written as things the reader can check or change (boundary choice, mesh,
  vacuum size), not as legal hedging.
- **Unsupported is one sentence.** "SCGSim does not support curved geometry
  for AEDT." Do not explain the internal reason on a reader page.
- **Results are approximate.** Quote recorded reference values to three or four
  significant figures ("about 95 fF") and explain the model settings. Add an
  agreement tolerance only when a measured comparison supports it. Full-precision
  values belong in the linked data files.

Examples from the existing docs:

| Before | After |
|---|---|
| "The dimensions and stack are model inputs, not measured capacitance or an accuracy claim." | "The capacitance you compute describes this geometry and this boundary choice." |
| "This page makes no claim of a completed native v3 body run." | Remove. If Q3D is not ready for the tutorial, leave it out of the tutorial. |
| "Preparation does not start the Palace solver." | "SCGSim does not start Palace for you. It writes a run directory you run yourself." |
| "These observations do not establish mesh convergence or numerical accuracy." | "If $C_{11}$ still changes between the last passes, raise `amr_max_passes`." |

## 5. Vocabulary

SCGSim has its own terms. A reader page must either avoid a term or define
it in one plain sentence the first time it appears on that page, then link
to the glossary or Concept page that owns it.

| Term | Plain definition to use |
|---|---|
| Entity | A named shape in the component |
| Net | A group of Entities that form one electrical conductor |
| GeometryPlan | The object where you assign Entities to Nets and add vacuum |
| Snapshot | The frozen result of `GeometryPlan.prepare()`; prepare again after changes |
| Handoff / run package | A self-contained directory with mesh, config and launch script that you run on the solver host |
| `handoff_id` | An ID that confirms loaded results came from a specific run package |
| Resolve | Load and check a returned run directory |
| Route A / Route B | Palace geometry representations; define on the Concept page before using |

Do not invent new nouns for things that have ordinary names. "Run
directory" is better than "returned package custody".

## 6. Sentences and layout

- One idea per sentence. Split sentences that carry more than two technical
  facts.
- Use the active voice and address the reader as "you".
- Use present tense for what the software does.
- Keep paragraphs to five lines or fewer in source.
- Use at most one or two links per paragraph. Do not link the same page
  twice in a section.
- Give units with every number and every argument that has one
  (`padding` is in micrometres).
- Explain each non-obvious argument in a code block in the text right after
  it. Do not leave an argument for the reader to guess.
- Keep a consistent tone within a page: tutorials teach, references specify.

## 7. Code in pages

- Code in Tutorial and How-to pages must run as written against the current
  API, in order, given the prerequisites. Mark anything else clearly as
  conceptual.
- Check every function and argument name against `src/scgsim/` before you
  publish. Do not copy from older pages or notebooks without checking.
- Show only the arguments the reader needs. Defaults stay out unless the
  page explains them.
- When the API forces something surprising (for example
  `ElectrostaticSim.mesh()` requiring `set_surface_epr()`), say so in one
  line, say it does not affect the result if that is true, and report the
  API issue to the maintainers.
- Python first; the OrPen notebook is an alternative and gets one short
  section with a link.

## 8. External examples and versions

- Each example pins **one** OrPen SC PDK revision, given once in the link to
  its source. Do not repeat the hash in the text.
- The installation page says which SCGSim version the docs describe and how
  to install it. It does not catalogue every revision used by every example.
  Example-specific setup goes on the example's own page.
- Do not copy notebook content into SCGSim docs. Explain the SCGSim API; link
  the notebook.

## 9. Navigation

- Every reader page appears in `_quarto.yml` navigation. A page that is not
  in the navigation is either contributor material (link it from
  `MAINTAINING.md`) or should be deleted.
- Put each page in the section that matches its type (section 2).
- Each concept has one owning page. Others link to it; they do not restate
  it.

## 10. Checklist before you finish

- [ ] The page has one type and one job, stated in its first paragraph.
- [ ] A newcomer could follow it without knowing SCGSim's internal terms.
- [ ] No hashes, lifecycle states, evidence wording or historical paths in
      the body.
- [ ] Limits are in one section and say what the reader can check.
- [ ] Code matches `src/scgsim/` and runs in order.
- [ ] Numbers have units; reference results identify the model and comparison settings.
- [ ] The page is in `_quarto.yml` and renders with
      `quarto render <page>` without warnings.
