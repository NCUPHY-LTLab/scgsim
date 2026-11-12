# SCGSim code style

Write Python so a maintainer who knows Python and superconducting circuits can follow what a module owns and what its main operation does. Keep code changes small enough to review, and preserve the package's established behavior and source ownership.

## Read the code

1. **State responsibility.** Give each module a docstring that says what it owns. Put the main entry point first; make it read as named steps, then place helpers in the order those steps use them.
2. **Keep one abstraction level per function.** Aim for 80 lines or fewer. A function over 150 lines needs a nearby reason, such as an order-dependent geometry kernel.
3. **Split by responsibility.** Aim for files under 800 lines. Split when parts change for different reasons, not by alphabet or by whether they are functions or classes. Avoid fragmenting a cohesive operation across tiny files.
4. **Keep the main path legible.** Move repeated validation into a named validator. Leave checks that determine behavior visible where that behavior is chosen.
5. **Comment why.** Explain physics, geometry, external-tool behavior, or required call order. Do not paraphrase the code. If the reason is unknown, write `TODO(human): why ...` rather than guessing.
6. **Define domain terms.** Expand an abbreviation such as `MA`, `MS`, or `SA` where it is defined. Give an unfamiliar domain term one concise meaning; mark an unconfirmed meaning `TODO(human)`.
7. **Make data shapes explicit.** Use `TypedDict` when JSON should remain a dictionary and a dataclass when related data travels together with behavior. Replace positional tuples and repeated `isinstance` tuples with named records or names that state their meaning.
8. **Name purpose, not mechanism.** Keep standard-library and third-party names unchanged. Use a concise name for the responsibility around them.
9. **Keep helpers distinct unless they are truly equivalent.** Merge helpers only when their inputs, behavior, and purpose are the same. If they differ, name the difference.
10. **Use underscores to show file scope.** A leading underscore means the name is used only in that file. A name imported from another module has no leading underscore and has a docstring. This naming rule does not make the name a supported public API; supported APIs are the package's documented facades and contracts.
11. **Choose functions or a class by data flow.** Use a pipeline of functions when data moves through named steps. Use a class when the same state travels through several steps and naturally belongs together. Do not wrap a sequence in a class just to group functions.

A small pipeline can make the required order clear:

```python
def build(self) -> ConstructionPlanRecord:
    """Run planning stages in order and return their combined result."""
    self.plan_interfaces()
    self.plan_surfaces()
    self.plan_topology()
    self.plan_volumes()
    self.plan_cuts_and_tags()
    return self.assemble()
```

## Package boundaries

- Let import paths show responsibility, for example `scgsim.geometry.planning.conductors`.
- Keep dependencies genuinely one-way: higher-level orchestration may use lower-level models and operations; lower layers must not import their callers. Avoid sibling cycles.
- Keep imports at module scope. Use a lazy import only when it avoids loading an optional heavy dependency on a path that does not need it; explain that reason beside the import.
- Shared names use ordinary names and docstrings. File-local helpers use a leading underscore. Do not reach into another object's private attributes.

## Design principles

These three principles decide where a rule, a fact and a refusal live. Each is shown with one case from this repository.

1. **One implementation per rule.** Decide a rule in one place and call that place from everywhere that needs it. After the source stage, decide geometry on the model grid in exact arithmetic, never in floating point. *Example:* #185 found the rule "three arc points on one line define no circle" in two places: the construction binder (`scgsim.geometry.construction.curved_geometry.boundaries`) tests it on floats and SGB (`scgsim.geometry.sgb.prisms.arcs`) on exact fractions, so three points on one line could pass the first and be refused by the second.
2. **Carry facts forward as typed fields.** A later stage reads what an earlier stage knew from a field. It never derives the fact again, never guesses it, and never parses an id. *Example:* before sidecar v3 (#170), the meshing stage took a band member's ring and position out of its id with a regular expression. Sidecar v3 states them as typed references, `carrier.primitive` → `support_primitive.ring` → `support_ring`, and no reader parses an id.
3. **The check that detects a problem writes the refusal.** The refusal names the object, its layer, the place in µm and who must act, and links to its explanation, all from the same data the check detected the problem with. *Example:* since #184, `check_manifold` (`scgsim.geometry.sgb.derivation.contacts`) finds a pinch at a vertex of the slab arrangement and names the conductor or region from the wedges around that same vertex, not from a search of the footprints near it.

## Refactoring and compatibility

An internal readability refactor preserves behavior, call order, data formats and algorithms. Keep bug fixes separate from refactoring so each change has a clear review purpose.

A change that could break a public contract needs a Human decision before implementation. Present the old and proposed interfaces, why the change is needed, known callers, and a migration path. The Human decides each proposed break; do not assume all breaks are forbidden or approved.

## Checks and review

Run the repository's unified quality entry point:

```bash
python scripts/check_quality.py
```

It checks tracked Python files with Ruff formatting and lint rules, the readable-style checker (including cross-file private imports and length limits), Pyright in standard mode across `src` while retaining existing strict settings, and import-linter. Tool versions are pinned by the repository; use its configured environment rather than fetching a latest version ad hoc.

A length or private-import exception must use the marker the checker recognizes, with a non-empty reason, at the exact location below:

A pending migration, an ownership boundary, or an established name is unfinished work, not an exception. A private-import marker must explain why the final design needs that import; keep remaining naming changes in the task's issue list until their callers have migrated.

- Put `# long-function: <reason>` on the `def` line of a function over 150 lines.
- Put `# long-file: <reason>` in a standalone top-level comment within the file's first 10 lines when the file exceeds 800 lines.
- Put `# private-import: <reason>` on the import's starting line when a cross-file private import is necessary.

For example, `def build(self) -> Plan:  # long-function: keeps the order-dependent kernel together` gives the checker a marker and a reason. Place exceptions narrowly and explain the actual constraint; prose elsewhere in the file does not satisfy the check.

These checks cover mechanical rules, not whether the code tells a clear story, has the right responsibility boundaries, or uses accurate domain meanings. Review those points as a maintainer. Run `pytest` and any real native solver workflow only when the task's authorization and scope require them. Unit-test doubles do not establish that a native solver workflow works.
