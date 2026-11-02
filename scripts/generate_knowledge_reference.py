"""Generate bilingual API pages and a source-bound public reference index.

The generator reads Python syntax and authored Markdown only. It never imports
SCGSim or initializes a solver.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

ROOT = Path(__file__).resolve().parents[1]
NAMESPACES = (
    "aedt",
    "aedt.specs",
    "geometry",
    "geometry.compiler",
    "geometry.models",
    "palace",
    "palace.mesh.quality",
    "semantics",
    "visualization",
)
INDEX_PATH = "docs/api/input-index.qmd"
DETAIL_PATH = "docs/api/analyze_epr.qmd"
CHINESE_INDEX_PATH = "zh-TW/docs/api/input-index.qmd"
CHINESE_DETAIL_PATH = "zh-TW/docs/api/analyze_epr.qmd"
USAGE_PATHS = (
    "docs/api/analyze-epr-usage.md",
    "zh-TW/docs/api/analyze-epr-usage.md",
)
WORKFLOW_PATH = "src/scgsim/aedt/epr/workflow.py"
ResourcePath: TypeAlias = str


@dataclass(frozen=True)
class _DetailPage:
    root: Path
    usage_path: str
    signature: str
    sources: list[str]
    index: str
    chinese: bool


def render(root: Path = ROOT) -> dict[ResourcePath, str]:
    """Read fixed API facades and authored inputs, then render four pages."""
    sources = _generation_sources(root)
    inventory = _public_documents(root)
    export_rows = _facade_exports(root)
    signature = _analyze_epr_signature(root)
    english_usage, chinese_usage = (_read_text(root, path) for path in USAGE_PATHS)

    english_index = _render_index(root, inventory, export_rows, sources, chinese=False)
    chinese_index = _render_index(root, inventory, export_rows, sources, chinese=True)
    english_detail = _render_detail(
        english_usage,
        _DetailPage(root, USAGE_PATHS[0], signature, sources, english_index, False),
    )
    chinese_detail = _render_detail(
        chinese_usage,
        _DetailPage(root, USAGE_PATHS[1], signature, sources, chinese_index, True),
    )
    return {
        INDEX_PATH: english_index,
        DETAIL_PATH: english_detail,
        CHINESE_INDEX_PATH: chinese_index,
        CHINESE_DETAIL_PATH: chinese_detail,
    }


def _generation_sources(root: Path) -> list[str]:
    sources = [
        "scripts/generate_knowledge_reference.py",
        "docs/knowledge-sources.json",
        WORKFLOW_PATH,
        *USAGE_PATHS,
        *(
            f"src/scgsim/{namespace.replace('.', '/')}/__init__.py"
            for namespace in NAMESPACES
        ),
    ]
    for source in sources:
        _read_bytes(root, source)
    return sources


def _public_documents(root: Path) -> list[str]:
    inventory = json.loads(_read_text(root, "docs/knowledge-sources.json"))
    documents = inventory["public_documents"]
    if not isinstance(documents, list) or any(
        not isinstance(path, str) for path in documents
    ):
        raise ValueError("public_documents must be a list of paths")
    if documents != sorted(set(documents)):
        raise ValueError("Public document inventory must be unique and sorted")
    for path in documents:
        _read_bytes(root, path)
    return documents


def _facade_exports(root: Path) -> dict[str, list[str]]:
    exports = {
        namespace: _declared_exports(root, namespace) for namespace in NAMESPACES
    }
    return exports


def _declared_exports(root: Path, namespace: str) -> list[str]:
    """Read literal __all__ assignments without importing a package facade."""
    source = f"src/scgsim/{namespace.replace('.', '/')}/__init__.py"
    tree = ast.parse(_read_text(root, source), filename=source)
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            names = _literal_names(node.value, namespace)
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            names.extend(_literal_names(node.value, namespace))
    if not names:
        raise ValueError(f"Explicit string __all__ required: scgsim.{namespace}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate __all__ entries: scgsim.{namespace}")
    return sorted(names)


def _literal_names(node: ast.expr, namespace: str) -> list[str]:
    value = ast.literal_eval(node)
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(name, str) for name in value
    ):
        raise ValueError(f"String-list __all__ required: scgsim.{namespace}")
    return list(value)


def _analyze_epr_signature(root: Path) -> str:
    tree = ast.parse(_read_text(root, WORKFLOW_PATH), filename=WORKFLOW_PATH)
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "analyze_epr"
        ),
        None,
    )
    if function is None:
        raise ValueError("Public analyze_epr function was not found")
    arguments = function.args
    if arguments.posonlyargs or arguments.args or arguments.vararg or arguments.kwarg:
        raise ValueError("Review formatter when analyze_epr is no longer keyword-only")
    if function.returns is None:
        raise ValueError("analyze_epr needs a declared return annotation")

    parameters = []
    for argument, default in zip(
        arguments.kwonlyargs, arguments.kw_defaults, strict=True
    ):
        parameter = ast.unparse(argument)
        if default is not None:
            parameter += f" = {ast.unparse(default)}"
        parameters.append(f"    {parameter},")
    return (
        "def analyze_epr(\n    *,\n"
        + "\n".join(parameters)
        + (f"\n) -> {ast.unparse(function.returns)}: ...")
    )


def _render_index(
    root: Path,
    documents: list[str],
    exports: dict[str, list[str]],
    sources: list[str],
    *,
    chinese: bool,
) -> str:
    title = (
        "公開參考輸入與匯出清單"
        if chinese
        else "Public reference inputs and export inventory"
    )
    description = (
        "完整列出公開匯出名稱，以及未收錄於離線套件的來源文件。"
        if chinese
        else "Declared public facade names and source documents outside the offline bundle."
    )
    parts = [_header(title, description, "docs/api/input-index", "reference")]
    if chinese:
        parts.append(
            "本頁由 `scripts/generate_knowledge_reference.py` 產生；請修改正式來源後重建。\n\n"
            "本索引列出九個公開 facade 的宣告名稱；名稱清單不取代各 API 的參數與行為契約。\n\n"
            "## 宣告的公開匯出\n\n"
        )
    else:
        parts.append(
            "Generated by `scripts/generate_knowledge_reference.py`; edit its canonical inputs.\n\n"
            "This index lists names declared by nine public facades; it does not replace each API's parameter and behavior contract.\n\n"
            "## Declared public exports\n\n"
        )
    for namespace, names in exports.items():
        parts.append(
            f"### scgsim.{namespace}\n\n"
            + ", ".join(f"`{name}`" for name in names)
            + ".\n\n"
        )
    parts.append(
        (
            "## 產生來源\n\n| 來源 | SHA-256 |\n|---|---|\n"
            if chinese
            else "## Generation sources\n\n| Source | SHA-256 |\n|---|---|\n"
        )
    )
    for source in sources:
        parts.append(f"| `{source}` | `{_digest(root, source)}` |\n")
    parts.append("\n")
    unbundled = [
        path
        for path in documents
        if path != "README.md"
        and not (path.startswith("docs/") and path.endswith(".qmd"))
    ]
    if chinese:
        parts.append(
            "## 未收錄於離線套件的公開文件\n\n"
            "README 與英文 `docs/*.qmd` 會收錄於離線 bundle；以下文件僅保留在儲存庫。\n\n"
        )
    else:
        parts.append(
            "## Public documents outside the offline bundle\n\n"
            "The README and English `docs/*.qmd` pages are bundled; the following public documents remain in the repository.\n\n"
        )
    parts.extend(f"- `{path}`\n" for path in unbundled)
    return "".join(parts)


def _render_detail(usage: str, page: _DetailPage) -> str:
    title = (
        "analyze_epr：讀取已保存的 HFSS 場"
        if page.chinese
        else "analyze_epr: evaluate retained HFSS fields"
    )
    description = (
        "已保存場 EPR 的參數、最小呼叫與函式簽章。"
        if page.chinese
        else "Saved-field EPR parameters, a minimal call and the current function signature."
    )
    identifier = "docs/api/scgsim-aedt/analyze_epr"
    parts = [_header(title, description, identifier, "api"), usage.rstrip(), "\n\n"]
    parts.append(
        "## 函式簽章\n\n```python\n" if page.chinese else "## Signature\n\n```python\n"
    )
    parts.extend((page.signature, "\n```\n\n"))
    parts.append(
        "## 直接來源\n\n| 來源 | SHA-256 |\n|---|---|\n"
        if page.chinese
        else "## Direct sources\n\n| Source | SHA-256 |\n|---|---|\n"
    )
    for source in (WORKFLOW_PATH, page.usage_path):
        if source not in page.sources:
            raise ValueError(
                f"Reference source is missing from source inventory: {source}"
            )
        parts.append(f"| `{source}` | `{_digest(page.root, source)}` |\n")
    label = "完整產生索引" if page.chinese else "Complete generation index"
    index_path = CHINESE_INDEX_PATH if page.chinese else INDEX_PATH
    index_hash = hashlib.sha256(page.index.encode("utf-8")).hexdigest()
    parts.append(f"\n[{label}]({Path(index_path).name}), SHA-256 `{index_hash}`.\n")
    return "".join(parts)


def _header(title: str, description: str, identifier: str, kind: str) -> str:
    fields = (
        ("title", title),
        ("description", description),
        ("scgsim-id", identifier),
        ("scgsim-kind", kind),
        ("scgsim-status", "CONVERGING"),
    )
    values = "".join(
        f"{key}: {json.dumps(value, ensure_ascii=False)}\n" for key, value in fields
    )
    return f"---\n{values}---\n\n"


def _digest(root: Path, path: str) -> str:
    return hashlib.sha256(_read_bytes(root, path)).hexdigest()


def _read_text(root: Path, path: str) -> str:
    return _read_bytes(root, path).decode("utf-8")


def _read_bytes(root: Path, path: str) -> bytes:
    source = root / path
    if source.resolve() != source or not source.is_file():
        raise ValueError(f"Reference source must be a regular non-symlink file: {path}")
    return source.read_bytes()


def main() -> None:
    """Write generated pages, or report stale pages without changing them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    for relative, content in render().items():
        destination = ROOT / relative
        if arguments.check:
            if (
                not destination.is_file()
                or destination.read_text(encoding="utf-8") != content
            ):
                raise SystemExit(f"Generated reference is stale: {relative}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
