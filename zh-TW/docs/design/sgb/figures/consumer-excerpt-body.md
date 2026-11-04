```{.json filename="semantic_geometry_route_b.sidecar.json（節錄）"}
{
 "arrangement": {
  "arc_edges": 0,
  "max_arc_deviation_grid": 0.0,
  "max_snap_drift_grid": 0.0
 },
 "binding": {
  "ids_by_tag": {
   "0": ["v:0b075ec90c6426c9", "v:1441394c11c5a86e", "..."],
   "1": ["e:96097c2a7413a1df", "e:4e13462abbf1c19c", "..."],
   "2": ["f:120f732fa4504c88", "f:27ce1e55539dce2b", "..."],
   "3": ["vol:1a3729bef5cef625", "vol:7346ec1a4ff1ccf9"]
  },
  "rule": "planned id -> native tag: tag = 1 + position in ids_by_tag[dim] (TopExp::MapShapes pre-order of the written compound); never by geometry"
 },
 "expected": {
  "edge_vertices": [
   [2, 1],
   [2, 3],
   "..."
  ],
  "face_area_um2": [400.0, 10000.0, "..."],
  "face_edges": [
   [1, 2, 3, 4, 5, 6],
   [7, 8, 9, 10],
   "..."
  ],
  "solid_faces": [
   [1, 2, 3, 4, 5, 6, 7, 8, "..."],
   [2, 3, 4, 5, 7, 8, 9, 10, "..."]
  ],
  "solid_volume_um3": [100000.0, 882160.0],
  "vertices_um": [
   [40.0, 48.0, 0.0],
   [40.0, 40.0, 0.0],
   "..."
  ]
 },
 "faces": {
  "f:179b3dd8d7bb9c70": {
   "face_kind": "sidewall",
   "group": "MA::ground_0000::sidewall",
   "kind": "wall",
   "negative_volume": "vol:739993ecf3919fb3",
   "positive_volume": "vol:7346ec1a4ff1ccf9",
   "z_um": [0.0, 0.2]
  }
 },
 "lineage": {
  "carriers": {
   "pad_top/ring0/edge5@m1000.0": {
    "family": "pad_top",
    "kind": "family_member",
    "member": "pad_top/ring0/edge5"
   }
  },
  "edges": {
   "e:06120918c00195e2": ["pad_top/ring0/edge5@m1000.0"]
  },
  "faces": {
   "f:179b3dd8d7bb9c70": ["ground_0000", "substrate"]
  },
  "membership": {
   "f:179b3dd8d7bb9c70": {
    "ground_0000": 1.0,
    "substrate": 1.0
   }
  }
 },
 "slivers": {
  "max_arc_deviation_grid": 0.0,
  "max_snap_drift_grid": 0.0,
  "narrowest_face_um": 0.2,
  "shortest_edge_um": 0.2,
  "smallest_angle_deg": 90.0,
  "thin_slab_count": 0
 },
 "volumes": {
  "vol:1a3729bef5cef625": {
   "aedt_name": "substrate",
   "emitted": true,
   "entity": "substrate",
   "group": "substrate",
   "kind": "dielectric",
   "material": "Si",
   "net": null,
   "shells": 1,
   "volume_um3": 100000.0
  }
 }
}
```
