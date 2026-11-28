"""The standalone copies a handoff puts in the run folder: the receipt recorder and the convergence runner.

Each is a module of this package (:mod:`.returned_receipt`, :mod:`.convergence_runner`) that imports run-folder
contracts (:mod:`scgsim.contracts`).  A run folder has no SCGSim, so the copy carries each contract's text in
place of those imports: the contracts the module imports, and the contracts they import, are placed where its
first contract import was, each once and before the contracts that import it, between marker comments.  Each
contract's module docstring, its ``__future__`` import and its imports of other contracts are left out.  The
copy then imports only the standard library, and a reader of the source looks in one place for each rule.

The copy is generated when the handoff is prepared and never stored in the repository.  A name that two of
the joined parts both define would silently replace one with the other, so it is refused.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType

from scgsim.refusal import internal

_CONTRACTS = "scgsim.contracts"


def standalone_text(module: ModuleType) -> str:
    """The text of ``module`` with the text of every contract it needs in place of its contract imports."""
    source = Path(_file(module)).read_text(encoding="utf-8")
    imports = _contract_imports(ast.parse(source), module.__name__)
    if not imports:
        raise RuntimeError(
            internal(f"{module.__name__} imports no {_CONTRACTS} module to embed")
        )
    contracts = _in_dependency_order([node.module or "" for node in imports])
    parts = {name: _contract_body(name) for name in contracts}
    lines = source.splitlines(keepends=True)
    skipped = {
        number
        for node in imports
        for number in range(node.lineno, (node.end_lineno or node.lineno) + 1)
    }
    first = imports[0].lineno
    embedded = [
        line
        for name in contracts
        for line in (
            f"# ---- {name}, embedded when the module is copied into a run folder ----\n",
            *parts[name],
            f"# ---- end of {name} ----\n",
        )
    ]
    text = "".join(
        [
            *lines[: first - 1],
            *embedded,
            *(
                line
                for number, line in enumerate(lines, start=1)
                if number >= first and number not in skipped
            ),
        ]
    )
    _refuse_clashes(module.__name__, source, parts)
    return text


def write_standalone(module: ModuleType, path: Path) -> None:
    """Write the standalone copy of ``module`` at ``path``, replacing any earlier copy, and make it executable."""
    path.unlink(missing_ok=True)
    path.write_text(standalone_text(module), encoding="utf-8")
    path.chmod(0o755)


def _file(module: ModuleType) -> str:
    if module.__file__ is None:
        raise RuntimeError(internal(f"{module.__name__} has no source file to copy"))
    return module.__file__


def _contract_imports(tree: ast.Module, owner: str) -> list[ast.ImportFrom]:
    """The module's top-level ``from scgsim.contracts.<name> import ...`` statements; any other SCGSim import is refused."""
    imports: list[ast.ImportFrom] = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(
            alias.name.split(".")[0] == "scgsim" for alias in node.names
        ):
            raise RuntimeError(
                internal(f"{owner} must import contracts by 'from ... import' only")
            )
        if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith(
            "scgsim"
        ):
            continue
        if not (node.module or "").startswith(f"{_CONTRACTS}.") or any(
            alias.asname for alias in node.names
        ):
            raise RuntimeError(
                internal(
                    f"{owner} may import only {_CONTRACTS} modules, without renaming, to be copied into a "
                    f"run folder; it imports {node.module}"
                )
            )
        imports.append(node)
    return imports


def _in_dependency_order(names: list[str]) -> list[str]:
    """``names`` and the contracts they import, each after the contracts it imports, each once."""
    ordered: list[str] = []

    def visit(name: str) -> None:
        if name in ordered:
            return
        tree = ast.parse(Path(_file(importlib.import_module(name))).read_text("utf-8"))
        for node in _contract_imports(tree, name):
            visit(node.module or "")
        ordered.append(name)

    for name in names:
        visit(name)
    return ordered


def _contract_body(name: str) -> list[str]:
    """A contract's lines without its module docstring, ``__future__`` imports and imports of other contracts."""
    contract = Path(_file(importlib.import_module(name))).read_text(encoding="utf-8")
    tree = ast.parse(contract)
    skipped: set[int] = set()
    for index, node in enumerate(tree.body):
        is_docstring = (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        is_contract = isinstance(node, ast.ImportFrom) and (
            node.module or ""
        ).startswith(f"{_CONTRACTS}.")
        if (is_docstring or is_future or is_contract) and node.end_lineno is not None:
            skipped.update(range(node.lineno, node.end_lineno + 1))
    return [
        line
        for number, line in enumerate(contract.splitlines(keepends=True), start=1)
        if number not in skipped
    ]


def _refuse_clashes(owner: str, source: str, parts: dict[str, list[str]]) -> None:
    """Refuse a name defined in two parts of the copy: the later definition would replace the earlier one."""
    defined: dict[str, str] = {}
    sources = {name: "".join(lines) for name, lines in parts.items()}
    sources[owner] = source
    for part, source in sources.items():
        for name in _defined_names(ast.parse(source)):
            if name in defined:
                raise RuntimeError(
                    internal(
                        f"the standalone copy of {owner} defines {name!r} in {defined[name]} and in {part}; "
                        "rename one of them"
                    )
                )
            defined[name] = part


def _defined_names(tree: ast.Module) -> set[str]:
    """The names a module body defines by assignment, ``def`` or ``class`` (imports may repeat)."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names
