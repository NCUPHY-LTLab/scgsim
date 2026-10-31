# SCGSim code style

Goal: a person who knows Python and the physics can open a module and follow it
without reading every line. Spend reader attention on what the code decides, not
on how it checks its inputs.

Before committing:

```bash
uvx ruff@0.17.3 check src
uvx ruff@0.17.3 format --check src
```

## Rules

1. **Tell the story first.** The module docstring says what the module owns and
   what it does not. The entry point comes first and reads as a sequence of named
   steps. Helpers follow in the order the entry point calls them.
2. **One level of abstraction per function.** Aim for 80 lines or fewer. A function
   over 150 lines needs a stated reason, such as an order-dependent geometry kernel.
3. **Split files by responsibility.** Aim for 800 lines or fewer. A split separates
   what changes for different reasons, not functions by alphabet or type.
4. **Keep validation out of the main path.** Move repeated checks into a named
   validator, so a reader can skip it. Main-path checks that decide behavior stay
   visible.
5. **Comments say why.** Record physics, geometry and AEDT or Palace quirks, and why
   an order matters. Do not restate the code. If you do not know why, do not guess:
   write `TODO(human): why ...`.
6. **Expand terms where they are defined.** `MA`, `MS`, `SA`, `Route A` and the like
   get a one-line meaning at their definition. Mark any meaning you cannot confirm
   with `TODO(human)`.
7. **Prefer named shapes to magic keys.** Use `TypedDict` for JSON that stays a dict
   and a dataclass for data with its own behavior. Name repeated `isinstance` tuples
   by what they mean. Replace positional tuples with named records.
8. **Names state the purpose.** Prefer a short purpose name to a long description of
   the mechanism.
9. **One copy of each helper.** Merge copies only when they really behave the same.
   When they differ, the names must say how.

## What a refactor must not change

- The public API: each facade's `__all__`, `scgsim.geometry.compiler` and
  `python -m scgsim.aedt.run`.
- Keys in handoff, receipt and result JSON.
- Numbers, algorithms, and the order of AEDT, Palace and gmsh calls.

Found a bug while refactoring? Leave the behavior, and report the bug separately.
A fix goes in its own commit.

## Example

`src/scgsim/aedt/runtime/transaction.py` is the reference. Its `_execute` shows the
whole transaction in a few lines; the verification and receipt details live in
`handoff_checks.py` and `receipts.py`.
