"""Carry a prepared SGB model through a handoff: its four files in the cohort, its identity in the receipt.

A handoff is moved to the machine that runs AEDT, so the model it names must travel with it (issue #67).
:func:`attach_sgb_model` copies the STEP, the BREP it was written from, the sidecar and the guard into the
handoff's ``model/`` folder and returns the record's payload with that folder as its relative
``model_dir``; :func:`sgb_model_paths` lists those files for the handoff's canonical file set.  On the AEDT
machine :func:`bind_sgb_model` reads the payload back against the run directory and checks every file
again before anything is imported.

:func:`sgb_model_source` is what the handoff's source record binds of the model (its identity and the four
file hashes); :func:`check_sgb_model_source` compares a source record with the model and refuses a record
that does not carry them, which is how a handoff or saved solution of the native builder is refused
(decision 6).
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from scgsim.aedt.preparation.sgb_model import BOUND_FILES, PreparedSgbModel

SGB_MODEL_DIR = "model"
SOURCE_KEYS = ("sgb_model_sha256", "sgb_model_files")


def sgb_model_paths(model: PreparedSgbModel) -> list[str]:
    """The handoff-relative paths of the model's four files, in the order of :data:`BOUND_FILES`."""
    return [f"{SGB_MODEL_DIR}/{model.files[role]['path']}" for role in BOUND_FILES]


def attach_sgb_model(model: PreparedSgbModel, run_dir: Path) -> dict[str, Any]:
    """Copy the model's files into ``run_dir/model`` and return its payload with that relative folder.

    The files are checked against their recorded hashes before and after the copy, so a handoff never
    carries a model that changed since it was prepared.
    """
    model.verify_files()
    target = run_dir / SGB_MODEL_DIR
    target.mkdir(parents=True, exist_ok=False)
    for role in BOUND_FILES:
        shutil.copy2(model.path(role), target / model.files[role]["path"])
    replace(model, model_dir=target).verify_files()
    payload = model.to_payload()
    payload["model_dir"] = SGB_MODEL_DIR
    return payload


def bind_sgb_model(payload: Mapping[str, Any], run_dir: Path) -> PreparedSgbModel:
    """The model of a handoff payload, bound to its files under ``run_dir/model`` and checked again.

    A payload whose model folder is not the handoff's own ``model/`` is refused, so a handoff can only
    name files it carries.
    """
    if payload.get("model_dir") != SGB_MODEL_DIR:
        raise ValueError(
            f"a handoff's SGB model lives in its {SGB_MODEL_DIR!r} folder, not "
            f"{payload.get('model_dir')!r}"
        )
    model = PreparedSgbModel.from_payload(payload)
    bound = replace(model, model_dir=Path(run_dir) / SGB_MODEL_DIR)
    bound.verify_files()
    return bound


def sgb_model_source(model: PreparedSgbModel) -> dict[str, Any]:
    """What a handoff's source record binds of the model: its identity and each file's SHA-256."""
    return {
        "sgb_model_sha256": model.identity_sha256,
        "sgb_model_files": {
            role: model.files[role]["sha256"] for role in BOUND_FILES
        },
    }


def check_sgb_model_source(source: Mapping[str, Any], model: PreparedSgbModel) -> None:
    """Refuse a source record that does not bind exactly ``model`` (see :func:`sgb_model_source`).

    A record without the SGB keys was written for a natively built model, which this version does not run
    or read (issue #67 decision 6).
    """
    missing = [key for key in SOURCE_KEYS if key not in source]
    if missing:
        raise ValueError(
            "the handoff binds no SGB model (it has no "
            f"{', '.join(missing)}); a natively built AEDT model is not run or read since issue #67, "
            "prepare it again"
        )
    expected = sgb_model_source(model)
    for key in SOURCE_KEYS:
        if source[key] != expected[key]:
            raise ValueError(f"the handoff's {key} differs from its SGB model")
