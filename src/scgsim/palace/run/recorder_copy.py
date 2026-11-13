"""The standalone copy of the receipt recorder that a handoff puts in the run folder.

The recorder (:mod:`.returned_receipt`) imports the run-folder contract
(:mod:`scgsim.contracts.palace_run_folder`).  A run folder has no SCGSim, so the copy carries the contract's
text in place of that one import: the contract's module docstring and its ``__future__`` import, if any, are
left out, and the rest is placed where the import was, between two marker comments.  The copy then imports
only the standard library, and a reader of the source looks in one place for the layout.
"""

from __future__ import annotations

import ast
from pathlib import Path

from scgsim.contracts import palace_run_folder
from scgsim.geometry.models.refusals import internal

from . import returned_receipt

_CONTRACT = palace_run_folder.__name__


def standalone_recorder_text() -> str:
    """The text of the recorder with the run-folder contract in place of its import of the contract."""
    recorder = Path(returned_receipt.__file__).read_text(encoding="utf-8")
    contract = Path(palace_run_folder.__file__).read_text(encoding="utf-8")
    first, last = _contract_import_lines(recorder)
    lines = recorder.splitlines(keepends=True)
    embedded = [
        f"# ---- {_CONTRACT}, embedded when the recorder is copied into a run folder ----\n",
        *_contract_body(contract),
        f"# ---- end of {_CONTRACT} ----\n",
    ]
    return "".join([*lines[: first - 1], *embedded, *lines[last:]])


def write_standalone_recorder(path: Path) -> None:
    """Write the standalone recorder at ``path``, replacing any earlier copy, and make it executable."""
    path.unlink(missing_ok=True)
    path.write_text(standalone_recorder_text(), encoding="utf-8")
    path.chmod(0o755)


def _contract_import_lines(recorder: str) -> tuple[int, int]:
    """The first and last line of the recorder's one import of the contract; any other form is refused."""
    imports = [
        node
        for node in ast.parse(recorder).body
        if isinstance(node, ast.ImportFrom) and node.module == _CONTRACT
    ]
    if len(imports) != 1 or imports[0].end_lineno is None:
        raise RuntimeError(
            internal(
                f"the receipt recorder must import {_CONTRACT} in exactly one 'from ... import' statement"
            )
        )
    return imports[0].lineno, imports[0].end_lineno


def _contract_body(contract: str) -> list[str]:
    """The contract's lines without its module docstring and ``__future__`` imports."""
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
        if (is_docstring or is_future) and node.end_lineno is not None:
            skipped.update(range(node.lineno, node.end_lineno + 1))
    return [
        line
        for number, line in enumerate(contract.splitlines(keepends=True), start=1)
        if number not in skipped
    ]
