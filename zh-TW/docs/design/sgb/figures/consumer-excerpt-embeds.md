```{.json filename="semantic_geometry_route_b.sidecar.json（節錄）"}
{
 "embedded_faces": [
  {
   "face": "f:642c08b554cd584e",
   "group": "PORT::PORT_SHEET__P1__0000",
   "host_volume": "vol:9d66ce4f20452849"
  }
 ],
 "mesh_hints": [
  {
   "edges": ["e:36da9569a3e69f89", "e:b1fa3f35d39449f4", "e:0beeb07b9aa96902", "..."],
   "kind": "band_boundary",
   "native_tags": [52, 53, 54, "..."],
   "source": "pad_top"
  },
  {
   "edges": ["e:3104024764366180"],
   "kind": "refinement_curve",
   "native_tags": [92],
   "source": "ref"
  }
 ],
 "ports": [
  {
   "adjacent_volumes": ["vol:9d66ce4f20452849"],
   "direction": [1.0, 0.0, 0.0],
   "faces": ["f:642c08b554cd584e"],
   "group": "PORT::PORT_SHEET__P1__0000",
   "host_entity_ids": ["ground_0000", "pad"],
   "host_volumes": ["vol:9d66ce4f20452849"],
   "port_index": 1,
   "port_sheet_id": "PORT_SHEET__P1__0000",
   "source_layer": "9/0",
   "source_name": "P1",
   "target_layer": "metal",
   "z_um": 0.1
  }
 ],
 "refinement_edges": [
  {
   "edge": "e:3104024764366180",
   "family": "ref",
   "host_face": "f:7e8c90cb3e7cf1ae",
   "written_as_free_edge": true
  }
 ]
}
```
