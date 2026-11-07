"""``simulation_drc``: one test per rule on small hand-built components, and a clean layout that passes."""

from __future__ import annotations

import pytest
from sgb_inputs import electrode_source

from scgsim.geometry import LayoutCheckError, simulation_drc
from scgsim.geometry.drc.polygons import polygon_findings

pytest.importorskip("gdsfactory")

NETS = {"A": ("chip/pad",), "B": ("chip/electrode",)}


def check(monkeypatch, *, gap_um=10.0, edit=None, **options):
    """Run the check on the electrode scene; ``edit`` may change the component before it runs."""
    source = electrode_source(monkeypatch, gap_um=gap_um)
    if edit is not None:
        edit(source.component)
    options.setdefault("nets", NETS)
    options.setdefault("coupon_padding_um", 10.0)
    return simulation_drc(
        source.component,
        layer_stack=source.layer_stack,
        material_records=source.material_records,
        **options,
    )


def rules(report):
    """The rule of each finding, in report order."""
    return [finding.rule for finding in report.findings]


def skipped(report):
    """The reason each skipped rule did not run, by rule."""
    return {item.rule: item.reason for item in report.skipped}


def test_clean_component_is_ok(monkeypatch):
    """A clean component reports no finding and renders as ok."""
    report = check(monkeypatch)

    assert report.ok
    assert report.findings == ()
    report.raise_for_findings()
    assert report.render().startswith("layout check: ok")


def test_clean_component_with_margins_runs_the_band_rule(monkeypatch):
    """With margins, a clean component also runs and passes the band rule."""
    report = check(monkeypatch, inset_margins_um=(0.5, 1.0))

    assert report.ok
    assert "surface-epr-bands" not in skipped(report)


def test_bands_rule_is_skipped_without_margins(monkeypatch):
    """Without margins the band rule is skipped, which is neither a failure nor a pass."""
    report = check(monkeypatch)

    assert skipped(report) == {"surface-epr-bands": "inset_margins_um was not given"}
    assert report.ok  # skipped is not a failure, and is not a pass either


def test_rule_1_self_intersecting_polygon(monkeypatch):
    """A bow tie is reported with its layer and crossing point."""

    def bow_tie(component):
        component.add_polygon(((40, 0), (50, 10), (50, 0), (40, 10)), layer=(3, 0))

    report = check(monkeypatch, edit=bow_tie)

    (finding,) = [item for item in report.findings if item.rule == "polygon-validity"]
    assert finding.layer == (3, 0)
    assert finding.location_um == pytest.approx((45.0, 5.0))
    assert "Self-intersection" in finding.message


class DegenerateRings:
    """A component whose GDS holds rings that gdsfactory itself would clean up before writing."""

    name = "degenerate"

    def dup(self):
        """The ring is already flat; flattening a copy changes nothing."""
        return self

    def flatten(self):
        """Nothing to flatten: the cell has no references."""
        pass

    def write_gds(self, path):
        """Write the degenerate rings as they are, without gdsfactory's clean-up."""
        import gdstk  # noqa: PLC0415 -- same optional stack as the fixture

        cell = gdstk.Cell(self.name)
        cell.add(gdstk.Polygon([(0, 0), (5, 0), (10, 0)], layer=3))
        cell.add(
            gdstk.Polygon([(20, 0), (30, 0), (30, 0), (30, 10), (20, 10)], layer=4)
        )
        library = gdstk.Library(unit=1e-6, precision=1e-9)
        library.add(cell)
        library.write_gds(str(path))


def test_rule_1_zero_area_and_repeated_vertex():
    """A collapsed ring and a ring with a repeated vertex are each reported once."""
    findings = polygon_findings(DegenerateRings(), {(3, 0): ["chip/line"]})

    by_layer = {finding.layer: finding for finding in findings}
    assert by_layer[(3, 0)].message == "polygon has zero area"
    assert by_layer[(3, 0)].entity == "chip/line"
    assert by_layer[(3, 0)].location_um == (0.0, 0.0)
    assert "repeats a vertex" in by_layer[(4, 0)].message
    assert by_layer[(4, 0)].entity == "degenerate"
    assert by_layer[(4, 0)].location_um == (30.0, 0.0)


def test_rule_1_ignores_vertices_off_the_grid(monkeypatch):
    """The source stage rounds off-grid vertices, so a rotated instance alone is no finding."""

    def rotated(component):
        import gdsfactory as gf  # noqa: PLC0415 -- same optional stack as the fixture

        child = gf.Component()
        child.add_polygon(((0, 0), (7, 0), (7, 3), (0, 3)), layer=(3, 0))
        component.add_ref(child).rotate(30)

    report = check(monkeypatch, edit=rotated)

    assert not [item for item in report.findings if item.rule == "polygon-validity"]


def test_rule_2_semantic_contract_keys(monkeypatch):
    """Ground-contribution keys that differ from the contract are reported."""

    def wrong_keys(component):
        component.info["component_semantics"]["ground_plane_contributions"] = [
            {"local_id": "g", "plane_id": "P", "mask_layer": [1, 0]}
        ]

    report = check(monkeypatch, edit=wrong_keys)

    (finding,) = report.findings
    assert finding.rule == "semantic-contract"
    assert finding.entity == "chip/g"
    assert "missing keys ['layer', 'level', 'selector_point_um']" in finding.message
    assert "unexpected keys ['mask_layer']" in finding.message


def test_rule_2_unknown_top_level_key(monkeypatch):
    """An unknown top-level semantics key is reported."""

    def extra(component):
        component.info["component_semantics"]["nets"] = {}

    report = check(monkeypatch, edit=extra)

    (finding,) = report.findings
    assert finding.rule == "semantic-contract"
    assert finding.entity == "chip"
    assert "unknown component_semantics v2 fields" in finding.message


def test_rule_2_port_and_level_must_be_declared(monkeypatch):
    """A port or level the layer stack does not declare is reported."""

    def bad_records(component):
        semantics = component.info["component_semantics"]
        semantics["conductor_regions"][0]["level"] = "no_such_level"
        semantics["ports"] = [
            {"name": "missing_port", "layer": [1, 0], "target_layer": "metal"}
        ]

    report = check(monkeypatch, edit=bad_records)

    by_entity = {item.entity: item.message for item in report.findings}
    assert (
        "level 'no_such_level' is not a level of the layer stack"
        in by_entity["chip/pad"]
    )
    assert "missing_port" in by_entity["chip/missing_port"]


def test_rule_2_registered_instance_must_exist(monkeypatch):
    """A registered reference that is not placed in the component is reported."""
    source = electrode_source(monkeypatch)
    other = electrode_source(monkeypatch)
    import gdsfactory as gf  # noqa: PLC0415 -- optional stack, skipped above when absent

    child = gf.Component()
    child.info["component_semantics"] = {"schema_version": 2}
    stranger = other.component.add_ref(child)

    report = simulation_drc(
        source.component,
        layer_stack=source.layer_stack,
        material_records=source.material_records,
        instances={"chip/child": stranger},
        nets=NETS,
    )

    (finding,) = report.findings
    assert finding.rule == "semantic-contract"
    assert finding.entity == "chip/child"
    assert "not an occurrence in its declared parent" in finding.message


def test_rule_2_findings_skip_the_production_stages(monkeypatch):
    """Semantic findings skip rules 3 to 5 instead of letting them refuse the same record."""

    def extra(component):
        component.info["component_semantics"]["nets"] = {}

    report = check(monkeypatch, edit=extra, inset_margins_um=(0.5, 1.0))

    assert set(skipped(report)) == {
        "physical-interpretation",
        "sgb-structure",
        "surface-epr-bands",
    }
    assert all("semantic-contract" in reason for reason in skipped(report).values())


def test_rule_3_plan_refusal_is_a_finding_and_skips_the_rest(monkeypatch):
    """A plan refusal becomes a finding and the later rules are skipped."""
    report = check(
        monkeypatch, nets={"A": ("chip/no_such_entity",)}, inset_margins_um=(1.0,)
    )

    (finding,) = report.findings
    assert finding.rule == "physical-interpretation"
    assert "unknown Entity ID 'chip/no_such_entity'" in finding.message
    assert set(skipped(report)) == {"sgb-structure", "surface-epr-bands"}
    assert "physical interpretation refused" in skipped(report)["sgb-structure"]


def test_rule_3_names_the_entity_and_layer_of_a_source_refusal(monkeypatch):
    """A source refusal names its Entity and layer."""

    def unmatched_selector(component):
        region = component.info["component_semantics"]["conductor_regions"][0]
        region["geometry"] = {"selector_point_um": [500.0, 500.0]}

    report = check(monkeypatch, edit=unmatched_selector)

    (finding,) = report.findings
    assert finding.rule == "physical-interpretation"
    assert finding.entity == "chip/pad"
    assert finding.layer == (1, 0)
    assert "matched 0 fused polygons" in finding.message


def test_rule_4_sgb_refusal_carries_its_location(monkeypatch):
    """An SGB refusal carries the location the error reports."""
    report = check(monkeypatch, gap_um=-5.0, inset_margins_um=(0.5,))

    (finding,) = report.findings
    assert finding.rule == "sgb-structure"
    assert "different-Net conductors overlap" in finding.message
    assert finding.location_um == pytest.approx((17.5, 10.0))
    assert list(skipped(report)) == ["surface-epr-bands"]


def test_rule_4_touching_nets_name_the_contact_point(monkeypatch):
    """Two Nets that touch are reported at the contact point."""
    report = check(monkeypatch, gap_um=0.0)

    (finding,) = report.findings
    assert finding.rule == "sgb-structure"
    assert "different-Net contact" in finding.message
    assert finding.location_um == pytest.approx((20.0, 0.0))


def test_rule_4_slivers_only_count_against_the_callers_limit(monkeypatch):
    """Slivers are findings only under the limit the caller sets."""
    assert check(monkeypatch).ok
    narrow = check(monkeypatch, gap_um=0.5, min_feature_um=1.0)

    assert not narrow.ok
    assert {item.rule for item in narrow.findings} == {"sgb-structure"}
    assert all("under the 1.0 um limit" in item.message for item in narrow.findings)
    assert all(item.location_um is not None for item in narrow.findings)


def test_rule_5_band_margins_off_the_grid(monkeypatch):
    """Margins off the model grid are a band-rule finding."""
    report = check(monkeypatch, inset_margins_um=(0.0005,))

    (finding,) = report.findings
    assert finding.rule == "surface-epr-bands"


def test_raise_for_findings_lists_every_finding(monkeypatch):
    """raise_for_findings lists every finding in one error."""

    def two_defects(component):
        component.add_polygon(((40, 0), (50, 10), (50, 0), (40, 10)), layer=(3, 0))
        component.add_polygon(((60, 0), (70, 10), (70, 0), (60, 10)), layer=(4, 0))

    report = check(monkeypatch, edit=two_defects)

    with pytest.raises(LayoutCheckError) as raised:
        report.raise_for_findings()
    message = str(raised.value)
    assert "2 finding(s)" in message
    assert "layer 3/0" in message
    assert "layer 4/0" in message
