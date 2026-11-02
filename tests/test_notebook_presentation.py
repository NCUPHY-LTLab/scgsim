"""Check escaped notebook formatting and styling of real Plotly figures."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import pytest

from scgsim.presentation.notebook import (
    cards,
    checked_theme,
    details,
    display_text,
    json_details,
    section,
    selection_text,
    style_figure,
    table,
)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_styling_preserves_figure_and_global_template(theme: str) -> None:
    """Style one real figure while keeping traces and Plotly defaults intact."""
    figure = go.Figure(go.Scatter(x=[0, 1], y=[2, 3]))
    traces = figure.to_plotly_json()["data"]
    default_template = pio.templates.default

    styled = style_figure(figure, theme=theme, height=640)

    assert styled is figure
    assert figure.to_plotly_json()["data"] == traces
    assert figure.layout.height == 640
    assert figure.layout.paper_bgcolor == {"light": "#ffffff", "dark": "#0d1117"}[theme]
    assert pio.templates.default == default_template


@pytest.mark.parametrize("theme", ["", "Dark", "automatic", None, 0, [], {}])
def test_invalid_theme_fails_before_styling(theme: object) -> None:
    """Reject invalid presentation choices before mutating a real figure."""
    figure = go.Figure()
    before = figure.to_plotly_json()

    with pytest.raises(ValueError, match="theme must be 'light' or 'dark'"):
        style_figure(figure, theme=theme)

    assert figure.to_plotly_json() == before


def test_display_selections_distinguish_absence_empty_and_zero() -> None:
    """Keep configuration absence and numeric zero visually distinct."""
    assert display_text(None) == "unknown"
    assert display_text(None, unknown="unavailable") == "unavailable"
    assert display_text(0) == "0"
    assert display_text(1.23456789) == "1.23457"
    assert selection_text(None, absent=True) == "not requested"
    assert selection_text(None) == "all"
    assert selection_text([]) == "none"
    assert selection_text([0, 2]) == "0, 2"
    assert selection_text("ab") == "ab"
    assert selection_text(b"ab") == "b'ab'"


def test_sections_escape_titles_and_keep_trusted_body_markup() -> None:
    """Escape caller labels while preserving the explicitly trusted HTML body."""
    rendered = section('<title & "quoted">', "<p>body</p>", class_name='a"b')
    assert "&lt;title &amp; &quot;quoted&quot;&gt;" in rendered
    assert 'class="scgsim-notebook a&quot;b"' in rendered
    assert "<p>body</p>" in rendered
    assert "color-scheme:light dark" in rendered
    assert "background:#0d1117" in section("title", "body", theme="dark")
    assert details("<title>", "<p>body</p>", opened=True) == (
        "<details open><summary>&lt;title&gt;</summary><p>body</p></details>"
    )


def test_cells_and_json_escape_data_instead_of_rendering_markup() -> None:
    """Treat every card, table cell and serialized JSON value as detached data."""
    for rendered in [
        cards([("<label>", "<value>")]),
        table(["<label>"], [["<value>"]]),
        json_details("<label>", {"value": "<value>"}),
    ]:
        assert "<label>" not in rendered
        assert "<value>" not in rendered
        assert "&lt;label&gt;" in rendered
        assert "&lt;value&gt;" in rendered


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_checked_theme_keeps_the_selected_string(theme: str) -> None:
    """The checked theme remains the caller's original string object."""
    assert checked_theme(theme) is theme
