"""``GeometryPreview``, a loaded source with its four modes, and the ``GeometryPreviewArtifact`` each mode renders to (an image, or the reason it is unavailable)."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

from scgsim.geometry.models.refusals import Declared, internal
from scgsim.refusal import found_text

from .loaders.common import (
    artifact_matches,
    confined_artifact_path,
    file_sha256,
    read_json_record,
    receipt_authority_sha256,
    write_json_atomically,
)
from .rendering import (
    add_parts,
    configure_headless_rendering,
    interactive_legend,
    legend,
    render_mode,
)
from .scene import (
    FIXED_COLORS,
    MODE_TITLES,
    MODES,
    VIEWS,
    SceneMode,
    with_scene_palette,
)

_WHERE_AEDT_RUN = "the AEDT run folder: fetch it again, or run the AEDT handoff again"


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


class _WriteOutcome(NamedTuple):
    """What one manifest write records: the mode just rendered with its images, or the mode that failed."""

    rendered_mode: str | None
    artifacts: list[dict[str, Any]] | None
    failed_mode: str | None
    failure: str | None


@dataclass(frozen=True)
class GeometryPreviewArtifact:
    """One rendered contact sheet, or one explicit unavailable preview mode."""

    mode: str
    status: Literal["available", "unavailable"]
    contact_sheet: Path | None = None
    reason: str | None = None

    def _ipython_display_(self) -> None:
        try:
            from IPython.display import HTML, Image, display  # noqa: PLC0415 -- IPython is needed only to display an artifact.
        except ImportError:
            print(self)
            return
        if self.contact_sheet is not None:
            display(Image(filename=str(self.contact_sheet)))
        else:
            display(
                HTML(
                    "<div style='padding:0.8rem;border:1px solid #bbb'>"
                    f"<strong>{MODE_TITLES[self.mode]}: Unavailable</strong><br>"
                    f"{self.reason}</div>"
                )
            )


class GeometryPreview:
    """A bound geometry source with four explicit preview modes."""

    def __init__(  # noqa: PLR0913 - binds the complete source identity and mode set
        self,
        *,
        root: Path,
        backend: Literal["palace", "aedt"],
        source_hashes: Mapping[str, str],
        source_identity: Mapping[str, Any] | None = None,
        bind_aedt_receipt: bool = False,
        modes: Mapping[str, SceneMode],
    ) -> None:
        self.root = root.resolve()
        self.backend = backend
        self._source_hashes = dict(source_hashes)
        self._source_identity = dict(source_identity or {})
        self._bind_receipt = bind_aedt_receipt
        self._modes = {name: with_scene_palette(mode) for name, mode in modes.items()}
        if set(self._modes) != set(MODES):
            raise ValueError(
                internal(
                    "geometry preview requires exactly four named modes"
                    + ": "
                    + found_text(sorted(self._modes), list(MODES))
                )
            )
        self._manifest_path = self.root / "metadata/geometry_preview_manifest.json"

    def show_materials(self) -> GeometryPreviewArtifact:
        """Render the material scene as a contact sheet."""
        return self._show("materials")

    def show_boundaries(self) -> GeometryPreviewArtifact:
        """Render the boundary scene as a contact sheet."""
        return self._show("boundaries")

    def show_surface_epr(self) -> GeometryPreviewArtifact:
        """Render the surface-EPR scene as a contact sheet."""
        return self._show("surface_epr")

    def show_mesh(self) -> GeometryPreviewArtifact:
        """Render the mesh scene as a contact sheet."""
        return self._show("mesh")

    def explore(
        self,
        mode: Literal["materials", "boundaries", "surface_epr", "mesh"],
    ) -> Any:
        """Display one inline interactive semantic scene in a notebook."""
        if mode not in MODES:
            raise ValueError(f"unsupported geometry preview mode: {mode!r}")
        selected = self._modes[mode]
        if selected.unavailable_reason is not None:
            return GeometryPreviewArtifact(
                mode=mode,
                status="unavailable",
                reason=selected.unavailable_reason,
            )
        self._verify_sources()
        try:
            import pyvista as pv  # noqa: PLC0415 -- PyVista is only needed for interactive rendering.
            from IPython.display import HTML, display  # noqa: PLC0415 -- IPython is only needed for the interactive legend.
        except ImportError as exc:
            raise RuntimeError(
                "interactive geometry preview requires scgsim[visualization] "
                "inside a Jupyter notebook"
            ) from exc
        display(HTML(interactive_legend(mode, selected.parts)))
        configure_headless_rendering()
        plotter = cast(Any, pv.Plotter(notebook=True, off_screen=True))
        plotter.set_background("white")
        add_parts(plotter, selected.parts)
        plotter.add_axes(line_width=3, labels_off=False)
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.reset_camera_clipping_range()
        return plotter.show(jupyter_backend="html", return_viewer=True)

    def show_all_previews(self) -> None:
        """Render and display every available semantic preview mode."""
        results = (
            self.show_materials(),
            self.show_boundaries(),
            self.show_surface_epr(),
            self.show_mesh(),
        )
        try:
            from IPython.display import display  # noqa: PLC0415 -- IPython is optional; text output is used without it.
        except ImportError:
            for result in results:
                print(result)
        else:
            for result in results:
                display(result)

    def _show(self, mode_name: str) -> GeometryPreviewArtifact:
        mode = self._modes[mode_name]
        if mode.unavailable_reason is not None:
            return GeometryPreviewArtifact(
                mode=mode_name,
                status="unavailable",
                reason=mode.unavailable_reason,
            )
        self._verify_sources()
        try:
            contact_sheet, artifacts = render_mode(
                root=self.root, mode_name=mode_name, parts=mode.parts
            )
        except Exception as exc:
            self._write_manifest(failed_mode=mode_name, failure=str(exc))
            raise
        self._write_manifest(rendered_mode=mode_name, artifacts=artifacts)
        return GeometryPreviewArtifact(
            mode=mode_name, status="available", contact_sheet=contact_sheet
        )

    def _verify_sources(self) -> None:
        for relative, expected in self._source_hashes.items():
            path = confined_artifact_path(self.root, relative)
            if not path.is_file() or file_sha256(path) != expected:
                raise ValueError(f"geometry preview source changed: {relative}")
        expected_receipt = self._source_identity.get("receipt_authority_sha256")
        if expected_receipt is not None:
            receipt_path = self.root / "metadata/aedt_run_receipt.json"
            if (
                not receipt_path.is_file()
                or receipt_authority_sha256(receipt_path) != expected_receipt
            ):
                raise ValueError(
                    Declared(
                        "AEDT preview receipt authority changed", _WHERE_AEDT_RUN
                    ).refusal(
                        found_text(
                            receipt_authority_sha256(receipt_path)
                            if receipt_path.is_file()
                            else None,
                            expected_receipt,
                        )
                    )
                )

    def _write_manifest(
        self,
        *,
        rendered_mode: str | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        failed_mode: str | None = None,
        failure: str | None = None,
    ) -> None:
        previous: dict[str, Any] = {}
        if self._manifest_path.is_file():
            previous = read_json_record(self._manifest_path)
        header = self._manifest_header()
        # Earlier mode records still describe this preview only when every header field is unchanged.
        same_source = all(previous.get(key) == value for key, value in header.items())
        mode_records = dict(previous.get("modes", {})) if same_source else {}
        outcome = _WriteOutcome(rendered_mode, artifacts, failed_mode, failure)
        for name in MODES:
            mode_records[name] = self._mode_record(
                name, mode_records.get(name, {}), outcome
            )
        payload = {
            **header,
            "modes": mode_records,
            "simulation_status_affected": False,
            "identity_role": "preview receipt; excluded from solver and handoff identity",
        }
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomically(self._manifest_path, payload)
        if self._bind_receipt:
            self._bind_aedt_receipt(payload)

    def _manifest_header(self) -> dict[str, Any]:
        """The manifest fields that identify the source and the renderer, in the order the manifest writes them."""
        return {
            "schema_version": "scgsim.geometry-preview.v1",
            "backend": self.backend,
            "source_hashes": self._source_hashes,
            "source_identity": self._source_identity,
            "renderer": {
                "name": "PyVista off-screen",
                "scgsim": _version("scgsim"),
                "pyvista": _version("pyvista"),
                "vtk": _version("vtk"),
                "meshio": _version("meshio") if self.backend == "palace" else None,
                "pillow": _version("pillow"),
            },
            "views": [item[0] for item in VIEWS],
            "palette": FIXED_COLORS,
            "tile_size_px": [1600, 1200],
            "contact_sheet_size_px": [4800, 4200],
        }

    def _mode_record(
        self, name: str, existing: Mapping[str, Any], outcome: _WriteOutcome
    ) -> dict[str, Any]:
        """One mode's record: just rendered, failed, unavailable, or available with its earlier images kept."""
        item = self._modes[name]
        if name == outcome.rendered_mode:
            return {
                "status": "available",
                "legend": legend(item.parts),
                "artifacts": outcome.artifacts,
            }
        if name == outcome.failed_mode:
            return {"status": "failed", "reason": outcome.failure}
        if item.unavailable_reason:
            return {"status": "unavailable", "reason": item.unavailable_reason}
        existing_artifacts = existing.get("artifacts", [])
        if any(
            not artifact_matches(self.root, artifact) for artifact in existing_artifacts
        ):
            existing_artifacts = []
        return {
            "status": "available",
            "legend": legend(item.parts),
            "artifacts": existing_artifacts,
        }

    def _bind_aedt_receipt(self, manifest: Mapping[str, Any]) -> None:
        receipt_path = self.root / "metadata/aedt_run_receipt.json"
        if not receipt_path.is_file():
            return
        receipt = read_json_record(receipt_path)
        artifacts = {
            item["path"]: item["sha256"]
            for mode in manifest["modes"].values()
            for item in mode.get("artifacts", ())
        }
        receipt["geometry_preview"] = {
            "status": "available" if artifacts else "not_rendered",
            "manifest": {
                "path": self._manifest_path.relative_to(self.root).as_posix(),
                "sha256": file_sha256(self._manifest_path),
            },
            "artifacts": artifacts,
            "simulation_status_affected": False,
        }
        write_json_atomically(receipt_path, receipt)
