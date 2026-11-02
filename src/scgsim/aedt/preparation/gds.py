"""Validate and rewrite GDS input for the native import layer map."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict, cast


class GdsPreflight(TypedDict):
    """The imported polygon layer map and non-polygon element counts."""

    polygon_layer_datatypes: set[tuple[int, int]]
    path_count: int
    label_count: int


def gds_preflight(path: Path) -> GdsPreflight:
    """Require nonempty polygonized GDS and report its exact layer pairs."""
    try:
        import gdstk  # noqa: PLC0415 -- optional GDS dependency is needed only for imported geometry.
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise RuntimeError(
            "GDS preflight requires the scgsim[aedt] gdstk dependency"
        ) from exc
    path_pairs = _raw_gds_path_datatypes(path)
    if path_pairs:
        raise ValueError(
            "GDS preflight found PATH layer/datatypes "
            f"{sorted(path_pairs)!r}; V1 requires polygonized CPW geometry"
        )
    library = gdstk.read_gds(path)
    path_pairs.update(
        (int(layer), int(datatype))
        for cell in cast(list[Any], library.cells)
        for path_item in cell.paths
        for layer, datatype in zip(path_item.layers, path_item.datatypes, strict=True)
    )
    if path_pairs:
        raise ValueError(
            "GDS preflight found PATH layer/datatypes "
            f"{sorted(path_pairs)!r}; V1 requires polygonized CPW geometry"
        )
    pairs = {
        (polygon.layer, polygon.datatype)
        for cell in cast(list[Any], library.cells)
        for polygon in cell.polygons
    }
    if not pairs:
        raise ValueError("GDS preflight found no polygons")
    return {
        "polygon_layer_datatypes": {
            (int(layer), int(datatype)) for layer, datatype in pairs
        },
        "path_count": 0,
        "label_count": sum(len(cell.labels) for cell in cast(list[Any], library.cells)),
    }


def _raw_gds_path_datatypes(path: Path) -> set[tuple[int, int]]:
    """Read PATH records because gdstk expands them to polygons on GDS import."""
    data = path.read_bytes()
    offset = 0
    in_path = False
    layer: int | None = None
    datatype: int | None = None
    result: set[tuple[int, int]] = set()
    while offset < len(data):
        if offset + 4 > len(data):
            raise ValueError("GDS preflight found a truncated record header")
        size = int.from_bytes(data[offset : offset + 2], "big")
        if size < 4 or offset + size > len(data):
            raise ValueError("GDS preflight found an invalid record length")
        record_type = data[offset + 2]
        payload = data[offset + 4 : offset + size]
        if record_type == 0x09:  # PATH
            in_path, layer, datatype = True, None, None
        elif in_path and record_type == 0x0D and len(payload) == 2:  # LAYER
            layer = int.from_bytes(payload, "big", signed=True)
        elif in_path and record_type == 0x0E and len(payload) == 2:  # DATATYPE
            datatype = int.from_bytes(payload, "big", signed=True)
        elif record_type == 0x11:  # ENDEL
            if in_path:
                if layer is None or datatype is None:
                    raise ValueError("GDS PATH record is missing layer or datatype")
                result.add((layer, datatype))
            in_path = False
        offset += size
    if in_path:
        raise ValueError("GDS preflight found an unterminated PATH record")
    return result


def write_import_gds(source: Path, destination: Path, effective_imports) -> None:
    """Lower explicit source pairs for numeric-layer-only native GDS import.

    Cells, references, database unit and polygon vertices stay source-authored;
    only the derived file's import selectors change. The source file is retained.
    """
    import gdstk  # noqa: PLC0415 -- optional GDS dependency is needed only for imported geometry.

    library = gdstk.read_gds(source)
    selectors = {
        (item["layer"], item["datatype"]): item["native_import_layer"]
        for item in effective_imports
    }
    for cell in cast(list[Any], library.cells):
        for polygon in cell.polygons:
            polygon.layer = selectors[(polygon.layer, polygon.datatype)]
            polygon.datatype = 0
    library.write_gds(destination, max_points=0)
