"""Regression coverage for generated Electrostatic config boundary bindings."""

from pathlib import Path

import pytest

from scgsim.palace.preparation.config import (
    TerminalBinding,
    build_electrostatic_config,
    configure_numerical_controls,
)


def test_electrostatic_config_keeps_nonempty_terminal_bindings() -> None:
    """Preserve terminal-boundary and index-map payloads from the old builder."""
    groups = {
        "boundary_surfaces": {
            "readout_contact": {
                "sgb_record": "final_physical_group",
                "source": "volume_interface",
                "solver_use": "solver_active",
                "interface_type": "MA",
                "conductor_component_id": "component-readout",
                "net_id": "net-readout",
                "phys_group": [17],
            }
        },
        "volumes": {
            "vacuum_domain": {
                "phys_group": [1],
                "physical_attribute": {"material_ids": ["vacuum"]},
                "source_provenance": {},
                "representation": "volume",
                "route": "A",
            }
        },
    }

    result = build_electrostatic_config(
        groups=groups,
        terminals=[TerminalBinding(name="readout", net_id="net-readout")],
        materials={
            "vacuum": {
                "kind": "vacuum",
                "permittivity": 1.0,
                "loss_tangent": 0.0,
            }
        },
        save_fields=0,
        unassigned_conductor_policy="error",
        exterior_boundary_policy="none",
        mesh_path=Path("palace.msh"),
        surface_epr_specs={},
        numerical=configure_numerical_controls(),
    )

    # These are the original generated values for the terminal config and index map.
    assert result.config["Boundaries"] == {
        "Terminal": [{"Index": 1, "Attributes": [17]}]
    }
    assert result.terminal_index_map == [
        {
            "section": "Boundaries.Terminal",
            "index": 1,
            "entry_name": "readout",
            "role": "terminal",
            "attributes": [17],
            "physical_names": ["readout_contact"],
            "metadata": {
                "net_id": "net-readout",
                "conductor_component_ids": ["component-readout"],
            },
            "terminal_name": "readout",
            "net_id": "net-readout",
            "conductor_component_ids": ["component-readout"],
            "terminal_attributes": [17],
        }
    ]
    assert result.ground_boundary_resolution == {
        "requested_net_ids": [],
        "physical_net_ids": [],
        "conductor_component_ids": [],
        "physical_names": [],
        "physical_attributes": [],
        "exterior_physical_names": [],
        "exterior_attributes": [],
        "union_attributes": [],
    }

    with pytest.raises(
        ValueError,
        match="SGB terminal 'readout' has no exact net match\\.",
    ):
        build_electrostatic_config(
            groups=groups,
            terminals=[TerminalBinding(name="readout", net_id="missing-net")],
            materials={
                "vacuum": {
                    "kind": "vacuum",
                    "permittivity": 1.0,
                    "loss_tangent": 0.0,
                }
            },
            save_fields=0,
            unassigned_conductor_policy="error",
            exterior_boundary_policy="none",
            mesh_path=Path("palace.msh"),
            surface_epr_specs={},
            numerical=configure_numerical_controls(),
        )
