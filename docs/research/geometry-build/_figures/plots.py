"""Figures, tables and quoted numbers of the "Geometry build performance" study.

Reads the measurements stored with the SGB design pages and writes, for both languages, the SVG figures and the Markdown
tables the study pages use.  Nothing is measured here and the site render never runs this file::

    python3 docs/research/geometry-build/_figures/plots.py

Only ``matplotlib`` is needed.  The editable data is the set of JSON files below (written by the design figure scripts
``why_scaling.py`` and ``emission.py``; the study's reproduction page lists the commands).  Outputs go to
``docs/research/geometry-build/figures/`` (English) and ``zh-TW/docs/research/geometry-build/figures/`` (Traditional Chinese),
with identical file names; ``numbers.json`` in each folder holds the values the prose quotes.

The work is split over ``plots_common`` (readers, palette and writers), ``plots_text`` (the labels of both languages), ``plots_scaling`` and
``plots_tables`` (the figures and tables).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plots_common import (
    OUTPUT,
    ROOT,
    SIZES,
    end_to_end,
    measured_rows,
    read,
    slope,
    slopes,
    tail_steps,
    validator_checks,
)
from plots_scaling import (
    api_comparison_figure,
    scaling_figure,
    scaling_tables,
    tail_figure,
)
from plots_tables import (
    emission_tables,
    occt_table,
    validation_figure,
    validation_tables,
)
from plots_text import CHECKS, S9_CHECKS, TEXT

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.fonttype": "none",
        "font.family": [
            "DejaVu Sans",
            "Noto Sans CJK JP",
        ],  # the second only supplies the Chinese glyphs for text layout
    }
)


# --------------------------------------------------------------------------- quoted numbers


def quoted_numbers(
    ones: dict[int, dict],
    timing: dict,
    oracle: dict,
    example: dict,
    validator: dict,
) -> dict:
    """The values the prose of the pages states, so a reviewer can compare the sentences with the data."""
    return {
        **_scaling_numbers(ones),
        **_tail_numbers(ones),
        **_api_numbers(timing, oracle),
        **_validation_numbers(timing, validator),
        "occt_all_ok_on_examples": example["occt"]["all_ok"],
    }


def _scaling_numbers(ones: dict[int, dict]) -> dict:
    """Derivation time, memory and face counts against N."""
    return {
        "derive_s": {n: ones[n]["build_wall_s"] for n in SIZES},
        "faces": {n: ones[n]["counts"]["faces"] for n in SIZES},
        "derive_slopes_against_n": [
            round(s, 2)
            for s in slopes(list(SIZES), [ones[n]["build_wall_s"] for n in SIZES])
        ],
        "rss_slopes_against_n": [
            round(s, 2)
            for s in slopes(list(SIZES), [ones[n]["peak_rss_mb"] for n in SIZES])
        ],
        "faces_slopes_against_n": [
            round(s, 2)
            for s in slopes(list(SIZES), [ones[n]["counts"]["faces"] for n in SIZES])
        ],
        "us_per_face": {
            n: round(ones[n]["build_wall_s"] / ones[n]["counts"]["faces"] * 1e6)
            for n in SIZES
        },
    }


def _tail_numbers(ones: dict[int, dict]) -> dict:
    """The steps from topology to a loadable model."""
    row = ones[20000]
    tail = tail_steps(row)
    return {
        "tail_s_at_20000": {k: round(v, 1) for k, v in tail.items()},
        "tail_total_one_process": round(
            tail["write"]
            + tail["reports"]
            + tail["sidecar"]
            + tail["load"]
            + tail["verify"],
            1,
        ),
        "tail_slopes_against_n": {
            k: [
                round(s, 2)
                for s in slopes(list(SIZES), [tail_steps(ones[n])[k] for n in SIZES])
            ]
            for k in ("write", "reports", "sidecar", "load", "verify")
        },
        "end_to_end_s": round(end_to_end(row, row["build_wall_s"]), 1),
        "sidecar_mb": {
            n: round(ones[n]["emission"]["file_bytes"]["sidecar"] / 1e6, 1)
            for n in SIZES
        },
    }


def _api_numbers(timing: dict, oracle: dict) -> dict:
    """The Gmsh-API oracle against the direct writer."""
    return {
        "api_to_direct_ratio": {
            h["n"]: round(
                h["api"]["total_s"]
                / (
                    h["direct"]["plan_s"]
                    + h["direct"]["render_brep_s"]
                    + h["direct"]["render_xao_s"]
                ),
                1,
            )
            for h in timing["head"]
        },
        "api_group_share": {
            h["n"]: round(h["api"]["physical_groups_s"] / h["api"]["total_s"], 3)
            for h in timing["head"]
        },
        "api_slopes_against_n": [
            round(s, 2)
            for s in slopes(
                [h["n"] for h in timing["head"]],
                [h["api"]["total_s"] for h in timing["head"]],
            )
        ],
        "oracle_runs_from_scaling_pass_s": {
            r["n_features"]: r["oracle_emit_s"] for r in oracle["rows"]
        },
    }


def _validation_numbers(timing: dict, validator: dict) -> dict:
    """The time of the structural checks against the number of faces."""
    checks = validator_checks(timing)
    faces = [
        r["counts"]["faces"] for r in sorted(timing["validator"], key=lambda r: r["n"])
    ]
    structural = [
        sum(v[i] for k, v in checks.items() if CHECKS[k][0] == "structural")
        for i in range(3)
    ]
    check_slopes = {
        k: round(slope(faces[0], v[0], faces[-1], v[-1]), 2) for k, v in checks.items()
    }
    return {
        "structural_total_s": [round(v, 2) for v in structural],
        "validator_cross_check_s9_total": {
            r["n_features"]: [
                r["validation_total_s"],
                round(sum(checks[k][i] for k in S9_CHECKS), 3),
            ]
            for i, r in enumerate(validator["rows"])
        },
        "structural_total_slopes_against_faces": [
            round(s, 2) for s in slopes(faces, structural)
        ],
        "check_slopes_against_faces": check_slopes,
        "check_slope_range_over_0.1s_at_largest": [
            min(s for k, s in check_slopes.items() if checks[k][-1] > 0.1),
            max(s for k, s in check_slopes.items() if checks[k][-1] > 0.1),
        ],
    }


# --------------------------------------------------------------------------- main


def main() -> None:
    """Write the figures, tables and quoted numbers for both languages."""
    measured = read("why-scaling-measured.json")
    timing = read("emission-timing-data.json")
    oracle = read("why-scaling-oracle.json")
    example = read("emission-example-numbers.json")
    ones = measured_rows(measured)
    numbers = quoted_numbers(
        ones, timing, oracle, example, read("why-validator-scaling.json")
    )
    for language, folder in OUTPUT.items():
        t = TEXT[language]
        scaling_figure(t, ones, folder)
        scaling_tables(language, ones, folder)
        api_comparison_figure(t, timing, folder)
        tail_figure(t, ones, folder)
        emission_tables(language, ones, timing, oracle, folder)
        validation_figure(t, timing, folder)
        validation_tables(language, timing, folder)
        occt_table(language, example, folder)
        (folder / "numbers.json").write_text(
            json.dumps(numbers, indent=1, sort_keys=True) + "\n"
        )
        print(f"wrote {folder.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
