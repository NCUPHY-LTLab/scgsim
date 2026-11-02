"""Headless and interactive rendering of detached visualization scenes."""

from __future__ import annotations

import html
import math
import os
import shutil
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .loaders.common import file_sha256
from .scene import MODE_TITLES, VIEWS, ScenePart


@dataclass(frozen=True)
class _ImageTools:
    """Optional Pillow operations needed only while assembling a sheet."""

    image: Any
    draw: Any
    font: Any


def render_mode(
    *, root: Path, mode_name: str, parts: tuple[ScenePart, ...]
) -> tuple[Path, list[dict[str, Any]]]:
    """Render all fixed views, build their contact sheet and record artifacts."""
    if not parts:
        raise ValueError(f"available preview mode {mode_name} has no geometry")
    try:
        # PyVista and Pillow are optional until a preview is actually rendered.
        import pyvista as pv  # noqa: PLC0415 - optional renderer, loaded on demand
        from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415 - optional renderer
    except ImportError as exc:
        raise RuntimeError("geometry rendering requires scgsim[visualization]") from exc

    final_dir = root / "previews/geometry" / mode_name
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{mode_name}-", dir=final_dir.parent
    ) as raw:
        temporary = Path(raw)
        center, span = _scene_camera(parts)
        configure_headless_rendering()
        _render_views(pv, temporary, parts, center, span)
        image_tools = _ImageTools(Image, ImageDraw, ImageFont)
        sheet = _build_contact_sheet(image_tools, temporary, mode_name, parts)
        sheet.save(temporary / "contact-sheet.png")
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.copytree(temporary, final_dir)

    artifacts: list[dict[str, Any]] = []
    for path in sorted(final_dir.glob("*.png")):
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return final_dir / "contact-sheet.png", artifacts


def _scene_camera(
    parts: tuple[ScenePart, ...],
) -> tuple[tuple[float, float, float], float]:
    bounds = _combined_bounds(parts)
    center = (
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2,
    )
    span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    return center, span


def _render_views(
    pyvista: Any,
    temporary: Path,
    parts: tuple[ScenePart, ...],
    center: tuple[float, float, float],
    span: float,
) -> None:
    for filename, label, direction, clip in VIEWS:
        plotter = cast(Any, pyvista.Plotter(off_screen=True, window_size=[1600, 1200]))
        plotter.set_background("white")
        add_parts(plotter, parts, clip=clip, center=center)
        plotter.add_axes(line_width=3, labels_off=False)
        plotter.add_text(label, position="upper_left", font_size=12, color="black")
        vector = _unit(direction)
        distance = max(span * 2.5, 1.0)
        plotter.camera.position = tuple(
            center[i] + vector[i] * distance for i in range(3)
        )
        plotter.camera.focal_point = center
        plotter.camera.up = _camera_up(vector)
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = max(span * 0.62, 1.0)
        plotter.reset_camera_clipping_range()
        plotter.screenshot(temporary / f"{filename}.png")
        plotter.close()


def _build_contact_sheet(
    image_tools: _ImageTools,
    temporary: Path,
    mode_name: str,
    parts: tuple[ScenePart, ...],
) -> Any:
    sheet = image_tools.image.new("RGB", (4800, 4200), "white")
    draw = image_tools.draw.Draw(sheet)
    title_font = _font(image_tools.font, 74)
    header_font = _font(image_tools.font, 42)
    body_font = _font(image_tools.font, 31)
    draw.text(
        (120, 70),
        f"{MODE_TITLES[mode_name]} — Geometry Preview",
        fill="black",
        font=title_font,
    )
    draw.text(
        (120, 170),
        "Color is presentation; semantic authority is solver metadata.",
        fill="#444444",
        font=body_font,
    )
    _draw_legend(draw, body_font, parts)
    draw.text(
        (120, 1360),
        "CURATED12 — cutaways are visualization clips, not solver boundaries",
        fill="#333333",
        font=header_font,
    )
    _paste_views(sheet, image_tools.image, temporary)
    return sheet


def _draw_legend(draw: Any, body_font: Any, parts: tuple[ScenePart, ...]) -> None:
    legend_entries = legend(parts)
    columns = 4
    rows = max(math.ceil(len(legend_entries) / columns), 1)
    row_height = min(220, 1080 // rows)
    for index, item in enumerate(legend_entries):
        column = index % columns
        row = index // columns
        x = 120 + column * 1140
        y = 270 + row * row_height
        draw.rectangle(
            (x, y + 8, x + 54, y + 62), fill=item["color"], outline="#333333"
        )
        wrapped = textwrap.wrap(item["label"], width=42) or [item["label"]]
        semantic = textwrap.wrap(item["semantic_id"], width=42) or [item["semantic_id"]]
        text = "\n".join([*wrapped, *semantic, f"{item['role']} · {item['count']}"])
        draw.multiline_text((x + 72, y), text, fill="black", font=body_font, spacing=2)


def _paste_views(sheet: Any, image_module: Any, temporary: Path) -> None:
    for index, (filename, _, _, _) in enumerate(VIEWS):
        tile = image_module.open(temporary / f"{filename}.png").convert("RGB")
        tile.thumbnail((1200, 900))
        x = (index % 4) * 1200
        y = 1500 + (index // 4) * 900
        sheet.paste(tile, (x, y))


def add_parts(
    plotter: Any,
    parts: tuple[ScenePart, ...],
    *,
    clip: tuple[str, float] | None = None,
    center: tuple[float, float, float] | None = None,
) -> None:
    """Add semantic scene datasets to a PyVista plotter."""
    for part in parts:
        dataset = part.dataset
        if clip is not None:
            if center is None:
                raise ValueError("visualization clip requires a scene center")
            normal = (1.0, 0.0, 0.0) if clip[0] == "x" else (0.0, 1.0, 0.0)
            dataset = dataset.clip(normal=normal, origin=center, invert=False)
        if dataset.n_cells:
            plotter.add_mesh(
                dataset,
                color=part.color,
                opacity=part.opacity,
                show_edges=part.show_edges,
                edge_color="#555555",
                line_width=0.5,
            )


def interactive_legend(mode_name: str, parts: tuple[ScenePart, ...]) -> str:
    """Format scene metadata as an escaped notebook legend table."""
    rows = "".join(
        "<tr>"
        f"<td><span style='display:inline-block;width:1.1rem;height:1.1rem;"
        f"background:{html.escape(item['color'])};border:1px solid #555'></span></td>"
        f"<td>{html.escape(item['label'])}</td>"
        f"<td>{html.escape(item['semantic_id'])}</td>"
        f"<td>{html.escape(item['role'])}</td>"
        f"<td style='text-align:right'>{item['count']}</td>"
        "</tr>"
        for item in legend(parts)
    )
    return (
        "<div style='margin:0.4rem 0 0.8rem'>"
        f"<strong>{html.escape(MODE_TITLES[mode_name])} — Interactive Geometry</strong>"
        "<div style='color:#555;margin:0.2rem 0 0.5rem'>"
        "Color is presentation; semantic authority is solver metadata."
        "</div>"
        "<table style='border-collapse:collapse'>"
        "<thead><tr><th></th><th>Label</th><th>Semantic ID</th><th>Semantic role</th>"
        "<th>Entities</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def legend(parts: tuple[ScenePart, ...] | list[ScenePart]) -> list[dict[str, Any]]:
    """Aggregate displayed entity counts by semantic identity and appearance."""
    aggregated: dict[tuple[str, str, str, str], int] = {}
    for part in parts:
        key = (part.semantic_id, part.label, part.role, part.color)
        aggregated[key] = aggregated.get(key, 0) + part.count
    return [
        {
            "semantic_id": semantic_id,
            "label": label,
            "role": role,
            "color": color,
            "count": count,
        }
        for (semantic_id, label, role, color), count in aggregated.items()
    ]


def _combined_bounds(parts: tuple[ScenePart, ...]) -> tuple[float, ...]:
    bounds = [part.dataset.bounds for part in parts if part.dataset.n_points]
    if not bounds:
        raise ValueError("preview geometry has no points")
    result = (
        min(item[0] for item in bounds),
        max(item[1] for item in bounds),
        min(item[2] for item in bounds),
        max(item[3] for item in bounds),
        min(item[4] for item in bounds),
        max(item[5] for item in bounds),
    )
    if not all(math.isfinite(value) for value in result):
        raise ValueError("preview geometry bounds are non-finite")
    return result


def configure_headless_rendering() -> None:
    """Select the available EGL device when rendering without a display."""
    if os.environ.get("DISPLAY"):
        return
    window_name = os.environ.setdefault(
        "VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow"
    )
    if window_name != "vtkEGLRenderWindow":
        return
    if "VTK_DEFAULT_EGL_DEVICE_INDEX" in os.environ:
        return
    try:
        from vtkmodules.vtkRenderingOpenGL2 import vtkEGLRenderWindow  # noqa: PLC0415 - EGL is optional until headless rendering
    except ImportError as exc:
        raise RuntimeError(
            "headless geometry preview requires VTK EGL support"
        ) from exc
    device_count = vtkEGLRenderWindow().GetNumberOfDevices()
    if device_count < 1:
        raise RuntimeError("headless geometry preview found no EGL rendering device")
    os.environ["VTK_DEFAULT_EGL_DEVICE_INDEX"] = str(device_count - 1)


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(item * item for item in vector))
    return tuple(item / length for item in vector)  # type: ignore[return-value]


def _camera_up(direction: tuple[float, float, float]) -> tuple[float, float, float]:
    return (0.0, 1.0, 0.0) if abs(direction[2]) > 0.9 else (0.0, 0.0, 1.0)


def _font(image_font: Any, size: int) -> Any:
    try:
        return image_font.truetype("DejaVuSans.ttf", size)
    except OSError:
        return image_font.load_default()
