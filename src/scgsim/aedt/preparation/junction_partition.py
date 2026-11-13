"""AEDT junction partitions: the shared source-domain rectangle algorithm under the AEDT record format."""

from dataclasses import replace

from scgsim.aedt.specs.epr import PlanarJunction
from scgsim.geometry.models.junctions import JunctionGeometry
from scgsim.geometry.planning.junction_planning.contact import JunctionContactRequest
from scgsim.geometry.planning.junction_planning.contact import (
    junction_contact_inventory as _shared_contact_inventory,
)
from scgsim.geometry.planning.junction_planning.core import (
    JunctionPartitionFormat,
    plain_record,
)
from scgsim.geometry.planning.junction_planning.partition import (
    partition_junctions as _partition_junctions,
)
from scgsim.geometry.planning.junction_planning.terminal import junction_terminal_line

METHOD = "scgsim.aedt.junction-partition.v1"
_FORMAT = JunctionPartitionFormat(
    metadata_key="aedt_junction_partitions",
    source_hash_key="aedt_authored_input_sha256",
    method=METHOD,
    id_prefix="aedt_junction",
    junction_metadata_key="aedt_junction_id",
    support_schema="scgsim.aedt.junction-support-plane.v1",
)


def junction_contact_inventory(
    build_input, source_polygon_id, terminal_a_net, terminal_b_net, stored_binding=None
):
    """Read source contacts using the AEDT support-plane schema."""
    return _shared_contact_inventory(
        build_input,
        JunctionContactRequest(
            source_polygon_id,
            terminal_a_net,
            terminal_b_net,
            stored_binding,
            support_schema=_FORMAT.support_schema,
        ),
    )


def junction_geometry(junction: PlanarJunction) -> JunctionGeometry:
    """Pass source geometry fields without solver inductance or capacitance."""
    return JunctionGeometry(
        junction_id=junction.junction_id,
        source_polygon_id=junction.source_polygon_id,
        terminal_a_net=junction.terminal_a_net,
        terminal_b_net=junction.terminal_b_net,
        direction_xy=junction.direction_xy,
        width_um=junction.width_um,
        metadata=junction.metadata,
    )


def partition_junctions(build_input, junctions, step):
    if not junctions:
        return build_input
    shared = build_input.metadata.get("junction_partitions")
    if shared is not None:
        records = {}
        for item in junctions:
            if item.junction_id not in shared:
                raise ValueError(
                    "junction is not present in the already-derived common model"
                )
            record = shared[item.junction_id]
            if record["authored_polygon_id"] != item.source_polygon_id or tuple(
                end["net_id"] for end in record["ends"]
            ) != (item.terminal_a_net, item.terminal_b_net):
                raise ValueError(
                    "AEDT junction source/terminal binding differs from common model"
                )
            junction_contact_inventory(
                build_input,
                item.source_polygon_id,
                item.terminal_a_net,
                item.terminal_b_net,
                item.metadata.get("junction_support_plane"),
            )
            central = next(
                polygon
                for polygon in build_input.polygons
                if polygon.polygon_id == record["central_polygon_id"]
            )
            junction_terminal_line(
                {"exterior": central.exterior, "holes": central.holes},
                item,
                record["z_um"],
            )
            adapted = plain_record(record)
            adapted["shared_geometry_method"] = record["method"]
            adapted["method"] = METHOD
            adapted["support_plane"]["schema_version"] = _FORMAT.support_schema
            records[item.junction_id] = adapted
        return replace(
            build_input,
            metadata={
                **build_input.metadata,
                "aedt_junction_partitions": records,
                "aedt_authored_input_sha256": build_input.metadata[
                    "junction_authored_input_sha256"
                ],
            },
        )
    neutral = tuple(
        JunctionGeometry(
            item.junction_id,
            item.source_polygon_id,
            item.terminal_a_net,
            item.terminal_b_net,
            item.direction_xy,
            item.width_um,
            item.metadata,
        )
        for item in junctions
    )
    return _partition_junctions(build_input, neutral, step, output_format=_FORMAT)
