"""Write the code map's stage tables from the docstrings of the package folders.

Every package folder under ``src/scgsim`` opens its ``__init__.py`` with one sentence; a stage folder adds
three lines, ``In:`` (what it takes, from whom), ``Out:`` (what it produces) and ``Next:`` (who reads that).
``AREAS`` lists every folder once, in the order the work flows, and says which folders are stages. This
script reads the docstrings (it never imports SCGSim) and writes one Markdown table per area for each
language edition of the site, which ``docs/code-map.qmd`` and its Chinese copy include. The English rows
are the docstrings; the Traditional Chinese rows are the authored translations in ``code_map_zh_tw.toml``
beside this script, each entry recording the English text it translates.

It refuses, naming the folder, a package folder ``AREAS`` does not list, a listed folder that is not a
package folder, a stage whose docstring lacks one of the three lines, a folder listed as something else
whose docstring has all three (list it as a stage), a folder without a translation entry, an entry for a
folder that is gone, and an entry whose recorded English is no longer the docstring's text (translate the
new text and record it).

    python scripts/generate_code_map.py          # write the tables
    python scripts/generate_code_map.py --check  # refuse tables that differ from the docstrings
"""

from __future__ import annotations

import argparse
import ast
import re
import tomllib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "src/scgsim"
LINE_KEYS = ("In", "Out", "Next")
TRANSLATION = "scripts/code_map_zh_tw.toml"


class Role(Enum):
    """What a folder is in the flow, which decides whether its docstring must carry the three lines."""

    STAGE = "stage"  # a stage of the work: its docstring has In, Out and Next
    AREA = "area"  # an area's own folder: its docstring describes the area as a whole
    PART = "part"  # one part of the stage named by ``part_of``: that stage's lines cover it
    SHARED = "shared"  # used by every area, outside the flow of any one of them


@dataclass(frozen=True)
class Folder:
    """One package folder, by its path below ``src/scgsim``, and its role."""

    path: str
    role: Role = Role.STAGE
    part_of: str = ""


@dataclass(frozen=True)
class Area:
    """One table of the code map: the folders of an area, in the order the work flows through them."""

    name: str
    folders: tuple[Folder, ...]


def _parts(stage: str, *names: str) -> tuple[Folder, ...]:
    return tuple(Folder(f"{stage}/{name}", Role.PART, stage) for name in names)


# The one statement of the order the work flows. A layout enters ``geometry/source``; the Palace route goes on
# through ``surface_epr``, ``sgb``, ``meshing`` and the Palace stages, the AEDT route through ``planning`` and
# the AEDT stages. ``contracts`` holds the files between stages.
AREAS = (
    Area(
        "geometry",
        (
            Folder("geometry", Role.AREA),
            Folder("geometry/models"),
            Folder("geometry/grid"),
            Folder("geometry/source"),
            Folder("geometry/interpretation"),
            Folder("geometry/junctions"),
            Folder("geometry/surface_epr"),
            Folder("geometry/planning"),
            *_parts(
                "geometry/planning",
                "contact_planning",
                "domain_planning",
                "surface_planning",
                "topology_planning",
            ),
        ),
    ),
    Area(
        "geometry/sgb",
        (
            Folder("geometry/sgb", Role.AREA),
            Folder("geometry/sgb/exact"),
            Folder("geometry/sgb/prisms"),
            Folder("geometry/sgb/arrangement"),
            *_parts("geometry/sgb/arrangement", "noding", "labelling", "derivation"),
            Folder("geometry/sgb/bands"),
            Folder("geometry/sgb/topology"),
            *_parts("geometry/sgb/topology", "assembly", "routes"),
            Folder("geometry/sgb/checks"),
            *_parts("geometry/sgb/checks", "validation", "slivers"),
            Folder("geometry/sgb/output"),
            *_parts(
                "geometry/sgb/output", "emission", "sidecar", "verification", "exchange"
            ),
            Folder("geometry/sgb/section"),
        ),
    ),
    Area(
        "meshing",
        (Folder("meshing"), *_parts("meshing", "recipes")),
    ),
    Area(
        "palace",
        (
            Folder("palace", Role.AREA),
            Folder("palace/simulation"),
            Folder("palace/source"),
            Folder("palace/mesh"),
            Folder("palace/config"),
            Folder("palace/run"),
            Folder("palace/results"),
            Folder("palace/reports"),
        ),
    ),
    Area(
        "aedt",
        (
            Folder("aedt"),
            Folder("aedt/specs"),
            Folder("aedt/preparation"),
            Folder("aedt/runtime"),
            Folder("aedt/runtime/families"),
            Folder("aedt/runtime/native"),
            Folder("aedt/epr"),
            Folder("aedt/results"),
            Folder("aedt/results/convergence"),
            Folder("aedt/presentation"),
        ),
    ),
    Area(
        "contracts",
        (Folder("contracts", Role.AREA), *_parts("contracts", "sgb_sidecar")),
    ),
    Area(
        "visualization",
        (Folder("visualization"), Folder("visualization/loaders")),
    ),
    Area(
        "shared",
        (
            Folder("", Role.SHARED),
            Folder("semantics", Role.SHARED),
            Folder("presentation", Role.SHARED),
        ),
    ),
)


@dataclass(frozen=True)
class Edition:
    """The words one language edition puts around the rows, the file it includes, and the language of its rows."""

    path: str
    header: tuple[str, str, str, str, str]
    shared_heading: str
    notes: dict[Role, str]
    language: (
        str  # "en": the docstring itself; otherwise the key of the authored translation
    )


EDITIONS = (
    Edition(
        "docs/_code-map-stages.md",
        ("Folder", "What it is", "In", "Out", "Next"),
        "Shared by every area",
        {
            Role.AREA: "The area's own folder: its docstring describes the area as a whole.",
            Role.PART: "Part of `{part_of}`; read it with that row.",
            Role.SHARED: "Used by every area, outside the flow of any one of them.",
        },
        "en",
    ),
    Edition(
        "zh-TW/docs/_code-map-stages.md",
        ("資料夾", "它是什麼", "In", "Out", "Next"),
        "各區共用",
        {
            Role.AREA: "這一區本身的資料夾：docstring 描述整個區。",
            Role.PART: "屬於 `{part_of}`，請與那一列一起閱讀。",
            Role.SHARED: "各區共用，不屬於任何一區的流程。",
        },
        "zh",
    ),
)


@dataclass(frozen=True)
class Docstring:
    """What the code map takes from one ``__init__.py``: its first sentence and its three lines."""

    summary: str
    lines: dict[str, str]

    def text(self, field: str) -> str:
        """The text of one field: ``summary`` or one of ``LINE_KEYS``."""
        return self.summary if field == "summary" else self.lines[field]


# folder (as the table names it) -> field -> {"en": the docstring text it translates, "zh": the translation}
Translations = dict[str, dict[str, dict[str, str]]]


def main() -> None:
    """Read the docstrings, refuse what the map cannot state, then write or check the tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    try:
        tables = render()
    except CodeMapError as error:
        raise SystemExit(str(error)) from None
    for relative, content in tables.items():
        destination = ROOT / relative
        if arguments.check:
            if (
                not destination.is_file()
                or destination.read_text(encoding="utf-8") != content
            ):
                raise SystemExit(
                    f"Code map tables are stale: {relative}; run python scripts/generate_code_map.py"
                )
            continue
        destination.write_text(content, encoding="utf-8")


class CodeMapError(ValueError):
    """A folder the code map cannot state as it is listed; the message names each folder."""


def render(root: Path = ROOT) -> dict[str, str]:
    """Return the text of each edition's include file, after checking every folder against ``AREAS``."""
    docstrings = read_docstrings(root)
    translations: Translations = tomllib.loads(
        (root / TRANSLATION).read_text(encoding="utf-8")
    )
    problems = check_folders(docstrings) or check_translations(docstrings, translations)
    if problems:
        raise CodeMapError("The code map cannot be generated:\n" + "\n".join(problems))
    return {
        edition.path: _render_edition(edition, docstrings, translations)
        for edition in EDITIONS
    }


def read_docstrings(root: Path) -> dict[str, Docstring | None]:
    """Every folder below ``src/scgsim`` that holds Python, with its parsed docstring or ``None``."""
    package = root / PACKAGE
    folders = {
        path.parent for path in package.rglob("*.py") if "__pycache__" not in path.parts
    }
    found: dict[str, Docstring | None] = {}
    for folder in sorted(folders):
        init = folder / "__init__.py"
        text = (
            ast.get_docstring(ast.parse(init.read_text(encoding="utf-8")))
            if init.is_file()
            else None
        )
        found[folder.relative_to(package).as_posix().removeprefix(".")] = (
            _parse(text) if text else None
        )
    return found


def _parse(text: str) -> Docstring:
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text.strip())]
    lines: dict[str, str] = {}
    for paragraph in paragraphs[1:]:
        # A key opens a field at the start of a paragraph or after a space; palace's folders put the three
        # lines in one paragraph, the others one per line.
        keys = list(re.finditer(rf"(?:^|(?<=\s))({'|'.join(LINE_KEYS)}):\s", paragraph))
        ends = [following.start() for following in keys[1:]] + [len(paragraph)]
        for key, end in zip(keys, ends[: len(keys)], strict=True):
            lines.setdefault(key.group(1), paragraph[key.end() : end].strip())
    return Docstring(paragraphs[0], lines)


def check_folders(docstrings: dict[str, Docstring | None]) -> list[str]:
    """One message per folder the listing and the docstrings disagree on."""
    listed = {folder.path: folder for area in AREAS for folder in area.folders}
    problems = [
        f"{_shown(path)}: a package folder AREAS does not list; add it where the work reaches it"
        for path in docstrings
        if path not in listed
    ]
    for path, folder in listed.items():
        problems.extend(
            _folder_problems(folder, path in docstrings, docstrings.get(path), listed)
        )
    return problems


def _folder_problems(
    folder: Folder, exists: bool, docstring: Docstring | None, listed: dict[str, Folder]
) -> list[str]:
    where = _shown(folder.path)
    if not exists:
        return [f"{where}: listed in AREAS but not a package folder"]
    if docstring is None:
        return [f"{where}/__init__.py: no docstring; it opens with one sentence"]
    missing = [key for key in LINE_KEYS if not docstring.lines.get(key)]
    if folder.role is Role.STAGE and missing:
        return [
            f"{where}/__init__.py: a stage docstring lacks its {', '.join(k + ':' for k in missing)} line"
        ]
    if folder.role is not Role.STAGE and not missing:
        return [
            f"{where}: listed as {folder.role.value}, but its docstring has In, Out and Next; list it as a stage"
        ]
    if folder.role is Role.PART and folder.part_of not in listed:
        return [f"{where}: part of {folder.part_of}, which AREAS does not list"]
    return []


def _shown(path: str) -> str:
    return f"{PACKAGE}/{path}".rstrip("/")


def check_translations(
    docstrings: dict[str, Docstring | None], translations: Translations
) -> list[str]:
    """One message per folder whose translation entry is missing, left over or translates other text."""
    listed = {_name(folder): folder for area in AREAS for folder in area.folders}
    problems = [
        f'{TRANSLATION}: entry ["{name}"] names no listed folder; remove it'
        for name in translations
        if name not in listed
    ]
    for name, folder in listed.items():
        docstring = docstrings[folder.path]
        assert docstring is not None  # check_folders refused a folder without one
        problems.extend(
            _entry_problems(name, folder, docstring, translations.get(name))
        )
    return problems


def _entry_problems(
    name: str,
    folder: Folder,
    docstring: Docstring,
    entry: dict[str, dict[str, str]] | None,
) -> list[str]:
    where = f'{_shown(folder.path)} ({TRANSLATION} ["{name}"])'
    if entry is None:
        return [
            f"{where}: no translation entry; add summary.en and summary.zh, and In, Out and Next for a stage"
        ]
    fields = _fields(folder)
    problems = [
        f"{where}: field {field} is not shown for this folder; remove it"
        for field in entry
        if field not in fields
    ]
    for field in fields:
        recorded = entry.get(field, {})
        if not recorded.get("zh"):
            problems.append(f"{where}: {field}.zh is missing")
        if recorded.get("en") != docstring.text(field):
            problems.append(
                f"{where}: the docstring's {field} is not the English this entry translates; set {field}.en to"
                f" the text below and translate it again in {field}.zh:\n    {docstring.text(field)}"
            )
    return problems


def _fields(folder: Folder) -> tuple[str, ...]:
    return ("summary", *LINE_KEYS) if folder.role is Role.STAGE else ("summary",)


def _name(folder: Folder) -> str:
    return folder.path or "scgsim"


def _render_edition(
    edition: Edition,
    docstrings: dict[str, Docstring | None],
    translations: Translations,
) -> str:
    sections = [
        f"<!-- Generated by scripts/generate_code_map.py from the package docstrings and {TRANSLATION}; do not edit. -->"
    ]
    for area in AREAS:
        heading = edition.shared_heading if area.name == "shared" else f"`{area.name}`"
        rows = [
            _row(folder, docstrings[folder.path], edition, translations)
            for folder in area.folders
        ]
        sections.append(f"### {heading}\n\n" + _table(edition.header, rows))
    return "\n\n".join(sections) + "\n"


def _row(
    folder: Folder,
    docstring: Docstring | None,
    edition: Edition,
    translations: Translations,
) -> tuple[str, ...]:
    assert docstring is not None  # check_folders refused a folder without one

    def text(field: str) -> str:
        if edition.language == "en":
            return _cell(docstring.text(field))
        return _cell(translations[_name(folder)][field][edition.language])

    cells = [text(field) for field in _fields(folder)]
    if folder.role is not Role.STAGE:
        cells += [edition.notes[folder.role].format(part_of=folder.part_of), "", ""]
    return (f"`{_name(folder)}`", *cells)


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    lines = [header, ("---",) * len(header), *rows]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines)


def _cell(text: str) -> str:
    """Docstring markup as Markdown: ``x`` and Sphinx roles become code spans; a pipe would split the cell."""
    text = re.sub(
        r":[a-z]+:`~?([^`]+)`", lambda m: f"`{m.group(1).rpartition('.')[2]}`", text
    )
    return text.replace("``", "`").replace("|", "\\|")


if __name__ == "__main__":
    main()
