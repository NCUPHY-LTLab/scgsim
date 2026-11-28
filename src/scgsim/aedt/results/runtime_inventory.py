"""The AEDT runtime source: which SCGSim source files an AEDT run is made of.

The list is computed on each call, never frozen: it is the static import
closure of every module under ``scgsim/aedt`` plus the two visualization
loaders that read AEDT results. Moving, renaming or deleting a file elsewhere
in ``src`` therefore changes the list instead of breaking AEDT preparation.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[3]
"""The directory that holds the ``scgsim`` package (``src`` in a checkout)."""

# The visualization loaders read AEDT receipts and result folders, and nothing
# under scgsim/aedt imports them, so the closure names them as roots.
_VISUALIZATION_ROOTS = (
    "src/scgsim/visualization/loaders/aedt.py",
    "src/scgsim/visualization/loaders/aedt_receipt.py",
)


def current_runtime_source_paths(source_root: Path = SOURCE_ROOT) -> tuple[str, ...]:
    """Return the sorted import closure of every AEDT module and its loaders."""
    aedt_modules = tuple(
        f"src/{path.relative_to(source_root).as_posix()}"
        for path in (source_root / "scgsim/aedt").rglob("*.py")
    )
    return runtime_import_closure(source_root, (*aedt_modules, *_VISUALIZATION_ROOTS))


def runtime_import_closure(
    source_root: Path, seed_paths: Iterable[str]
) -> tuple[str, ...]:
    """Return source paths reached by static internal Python imports."""
    return _RuntimeImportClosure(source_root).build(seed_paths)


class _RuntimeImportClosure:
    """Collect source modules reachable from one set of canonical source seeds."""

    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root.resolve()
        self.paths: set[str] = set()
        self.pending: list[Path] = []

    def build(self, seed_paths: Iterable[str]) -> tuple[str, ...]:
        """Add all seeds, walk their imports, and return sorted source paths."""
        for seed in seed_paths:
            self.add_seed(seed)
        while self.pending:
            self.expand(self.pending.pop())
        return tuple(sorted(self.paths))

    def add_seed(self, seed: str) -> None:
        """Validate and add one canonical source path as a closure root."""
        relative = Path(seed)
        if (
            relative.is_absolute()
            or not relative.parts
            or relative.parts[0] != "src"
            or ".." in relative.parts
        ):
            raise ValueError("runtime source seed must be a contained src path")
        path = self.source_root.joinpath(*relative.parts[1:])
        if not path.is_file():
            raise FileNotFoundError(f"runtime source seed is missing: {seed}")
        self.add_path(path)
        self.add_module(_module_name(path, self.source_root))

    def add_path(self, path: Path) -> None:
        """Schedule one source file once for import parsing."""
        canonical = f"src/{path.relative_to(self.source_root).as_posix()}"
        if canonical not in self.paths:
            self.paths.add(canonical)
            self.pending.append(path)

    def add_module(self, module: str) -> None:
        """Add an internal module plus every source package initializer."""
        parts = module.split(".")
        if not parts or parts[0] != "scgsim" or any(not part for part in parts):
            return
        for depth in range(1, len(parts)):
            package_init = self.source_root.joinpath(*parts[:depth], "__init__.py")
            if package_init.is_file():
                self.add_path(package_init)
        package_init = self.source_root.joinpath(*parts, "__init__.py")
        if package_init.is_file():
            self.add_path(package_init)
            return
        module_path = self.source_root.joinpath(*parts[:-1], f"{parts[-1]}.py")
        if module_path.is_file():
            self.add_path(module_path)

    def expand(self, path: Path) -> None:
        """Walk every import node, including imports inside lazy code paths."""
        module = _module_name(path, self.source_root)
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.add_import(node)
            elif isinstance(node, ast.ImportFrom):
                self.add_from_import(node, package)

    def add_import(self, node: ast.Import) -> None:
        """Resolve each absolute module in an import statement."""
        for alias in node.names:
            self.add_module(alias.name)

    def add_from_import(self, node: ast.ImportFrom, package: str) -> None:
        """Resolve a base module and any imported child modules."""
        base = _import_from_base(node, package)
        self.add_module(base)
        if base.startswith("scgsim"):
            for alias in node.names:
                if alias.name != "*":
                    self.add_module(f"{base}.{alias.name}")


def _module_name(path: Path, source_root: Path) -> str:
    relative = path.relative_to(source_root).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _import_from_base(node: ast.ImportFrom, package: str) -> str:
    """Resolve an absolute or relative ImportFrom base name."""
    base = node.module or ""
    if not node.level:
        return base
    package_parts = package.split(".") if package else []
    remove = node.level - 1
    if remove > len(package_parts):
        return ""
    prefix = ".".join(package_parts[: len(package_parts) - remove])
    return ".".join(part for part in (prefix, base) if part)
