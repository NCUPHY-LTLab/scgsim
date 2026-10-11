"""Pure canonical source-to-effective support projection for preparation and decoding."""

def _prepared_lumped_supports(supports, source):
    """Derive explicit support XY/Z from the same physical map as all bodies."""
    from scgsim.aedt.specs.modeling import PhysicalLayerSpec, _ZMap

    mapping = _ZMap(source["modeling"]["modeling"], tuple(
        PhysicalLayerSpec(**item) for item in source["modeling"]["source_layers"]
    ))
    conductors = source["conductors"]
    available = {item["semantic_id"] for item in conductors}
    available.update(item["source_semantic_id"] for item in conductors)
    records = []
    for support in supports:
        if not support.exterior:
            raise ValueError(f"support {support.support_id!r} has no resolved original source polygon")
        if support.contact_physical_layer_id not in mapping.layers:
            raise ValueError(f"support {support.support_id!r} references an undeclared contact physical layer")
        if not set((*support.terminal_a_entity_ids, *support.terminal_b_entity_ids)).issubset(available):
            raise ValueError(f"support {support.support_id!r} has missing conductor ownership")
        contacts = tuple(support.world_point(point) for point in support.contact_points_um)
        if support.source_z_um is None:
            raise ValueError("neutral support requires its original locator plane")
        if any(point[2] != support.source_z_um for point in contacts):
            raise ValueError("horizontal lumped support contacts differ from its original locator plane")
        layer = mapping.layers[support.contact_physical_layer_id]
        if not layer.z_min_um <= contacts[0][2] <= layer.z_max_um:
            raise ValueError("lumped support source contact plane is outside its declared physical layer")
        a, b, c, d, _, _ = support.source_transform
        nx, ny, nz = support.normal
        normal = (a * nx + b * ny, c * nx + d * ny, nz)
        records.append({"support_id": support.support_id, "source": support.to_payload(),
                        "effective": {"exterior": [list(support.world_point(point)) for point in support.exterior],
                                      "holes": [[list(support.world_point(point)) for point in ring] for ring in support.holes],
                                      "normal": list(normal), "z_um": mapping.map_z(support.source_z_um),
                                      "contact_points_um": [[x, y, mapping.map_z(z)] for x, y, z in contacts],
                                      "contact_physical_layer_id": support.contact_physical_layer_id}})
    if len({item["support_id"] for item in records}) != len(records):
        raise ValueError("neutral support IDs must be unique")
    return records

