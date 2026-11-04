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
    SCHEDULES,
    SIZES,
    end_to_end,
    one_process_rows,
    read,
    schedule_rows,
    slope,
    slopes,
    tail_steps,
    validator_checks,
)
from plots_scaling import (
    api_comparison_figure,
    parallel_figure,
    parallel_tables,
    scaling_figure,
    scaling_tables,
    speedups,
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


def quoted_numbers(  # noqa: PLR0913 -- the six measured inputs, each read from its own file
    by_schedule: dict,
    ones: dict[int, dict],
    timing: dict,
    oracle: dict,
    example: dict,
    validator: dict,
) -> dict:
    """The values the prose of the pages states, so a reviewer can compare the sentences with the data."""
    return {
        **_scaling_numbers(ones),
        **_tail_numbers(ones, by_schedule),
        **_api_numbers(timing, oracle),
        **_validation_numbers(timing, validator),
        **_parallel_numbers(ones, by_schedule),
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


def _tail_numbers(ones: dict[int, dict], by_schedule: dict) -> dict:
    """The steps from topology to a loadable model, and what a faster derivation would gain end to end."""
    row = ones[20000]
    tail = tail_steps(row)
    base = end_to_end(row, row["build_wall_s"])
    best = by_schedule[(20000, 8, 8)]["build_wall_s"]
    limit = row["amdahl"]["amdahl_speedup_limit"]
    return {
        "tail_s_at_20000": {k: round(v, 1) for k, v in tail.items()},
        "tail_total_one_process": round(
            tail["write_w1"]
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
            for k in ("write_w1", "reports", "sidecar", "load", "verify")
        },
        "end_to_end_one_process_s": round(base, 1),
        "end_to_end_with_best_derivation_s": round(end_to_end(row, best), 1),
        "end_to_end_gain_best": round(base / end_to_end(row, best), 3),
        "end_to_end_gain_at_amdahl_limit": round(
            base / end_to_end(row, row["build_wall_s"] / limit), 3
        ),
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


def _parallel_numbers(ones: dict[int, dict], by_schedule: dict) -> dict:
    """Speed-up, the serial share of the stages, memory and identity across the schedules."""
    gain = speedups(by_schedule)
    workers_rss = {
        n: [by_schedule[(n, t, w)]["peak_worker_rss_mb"] for t, w in SCHEDULES if w > 1]
        for n in SIZES
    }
    return {
        "speedup_best": {
            n: round(max(v for (t, w), v in gain[n].items() if w > 1), 2) for n in SIZES
        },
        "speedup_one_worker_tiled": {
            n: [
                round(min(gain[n][(t, 1)] for t in (4, 8)), 2),
                round(max(gain[n][(t, 1)] for t in (4, 8)), 2),
            ]
            for n in SIZES
        },
        "serial_share_of_stages": {
            n: round(
                ones[n]["amdahl"]["serial_stages_s"]
                / (
                    ones[n]["amdahl"]["serial_stages_s"]
                    + ones[n]["amdahl"]["parallel_stages_wall_s"]
                ),
                3,
            )
            for n in SIZES
        },
        "amdahl_limit": {n: ones[n]["amdahl"]["amdahl_speedup_limit"] for n in SIZES},
        "worker_peak_rss_range_mb": {
            n: [min(v), max(v)] for n, v in workers_rss.items()
        },
        "parent_peak_rss_20000_mb": {
            f"{t}x{t}/{w}": by_schedule[(20000, t, w)]["peak_rss_mb"]
            for t, w in SCHEDULES
        },
        "distinct_identity_digests": {
            n: len({by_schedule[(n, t, w)]["identity_digest"] for t, w in SCHEDULES})
            for n in SIZES
        },
        "all_validation_passed": all(
            r["validation_passed"] for r in by_schedule.values()
        ),
    }


# --------------------------------------------------------------------------- main


def main() -> None:
    """Write the figures, tables and quoted numbers for both languages."""
    measured = read("why-scaling-measured.json")
    timing = read("emission-timing-data.json")
    oracle = read("why-scaling-oracle.json")
    example = read("emission-example-numbers.json")
    ones = one_process_rows(measured)
    by_schedule = schedule_rows(measured)
    numbers = quoted_numbers(
        by_schedule, ones, timing, oracle, example, read("why-validator-scaling.json")
    )
    for language, folder in OUTPUT.items():
        t = TEXT[language]
        scaling_figure(t, ones, by_schedule, folder)
        scaling_tables(language, ones, folder)
        api_comparison_figure(t, timing, folder)
        tail_figure(t, ones, folder)
        emission_tables(language, ones, timing, oracle, folder)
        validation_figure(t, timing, folder)
        validation_tables(language, timing, folder)
        occt_table(language, example, folder)
        parallel_figure(t, by_schedule, folder)
        parallel_tables(language, by_schedule, ones, folder)
        (folder / "numbers.json").write_text(
            json.dumps(numbers, indent=1, sort_keys=True) + "\n"
        )
        print(f"wrote {folder.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
