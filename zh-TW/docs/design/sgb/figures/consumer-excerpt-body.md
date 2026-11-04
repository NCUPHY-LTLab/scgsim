```{.json filename="semantic_geometry_route_b.sidecar.json（節錄）"}
{
 "arrangement": {
  "arc_edges": 0,
  "max_arc_deviation_grid": 0.0,
  "max_snap_drift_grid": 0.0
 },
 "binding": {
  "ids_by_tag": {
   "0": ["v:889c0afb53729cd6", "v:5ae72f37e0d2bca6", "..."],
   "1": ["e:f1a409262d686007", "e:717d8fb4f6b8f22d", "..."],
   "2": ["f:27ce1e55539dce2b", "f:487ca41cd08e5769", "..."],
   "3": ["vol:1a3729bef5cef625", "vol:9d66ce4f20452849"]
  },
  "rule": "planned id -> native tag: tag = 1 + position in ids_by_tag[dim] (TopExp::MapShapes pre-order of the written compound); never by geometry"
 },
 "expected": {
  "edge_vertices": [
   [2, 1],
   [2, 3],
   "..."
  ],
  "face_area_um2": [10000.0, 400.0, "..."],
  "face_edges": [
   [1, 2, 3, 4],
   [5, 6, 7, 8, 9, 10],
   "..."
  ],
  "solid_faces": [
   [1, 2, 3, 4, 5, 6, 7, 8],
   [1, 4, 5, 6, 7, 8, 9, 10, "..."]
  ],
  "solid_volume_um3": [100000.0, 882160.0],
  "vertices_um": [
   [0.0, 100.0, -10.0],
   [0.0, 0.0, -10.0],
   "..."
  ]
 },
 "faces": {
  "f:179b3dd8d7bb9c70": {
   "face_kind": "sidewall",
   "group": "MA::ground_0000::sidewall",
   "kind": "wall",
   "negative_volume": "vol:f8f1d1c3d3854405",
   "positive_volume": "vol:9d66ce4f20452849",
   "z_um": [0.0, 0.2]
  }
 },
 "lineage": {
  "carriers": {
   "PORT_SHEET__P1__0000/polygon0/ring0/edge4": {
    "edge": 4,
    "entity": "PORT_SHEET__P1__0000",
    "kind": "polygon_edge",
    "polygon": 0,
    "ring": 0
   }
  },
  "edges": {
   "e:02213c44b2348835": ["PORT_SHEET__P1__0000/polygon0/ring0/edge4"]
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
  "narrowest_face_um": 0.1,
  "shortest_edge_um": 0.1,
  "smallest_angle_deg": 90.0,
  "thin_slab_count": 0
 },
 "volumes": {
  "vol:1a3729bef5cef625": {
   "aedt_name": "VOL__substrate",
   "emitted": true,
   "entity": "substrate",
   "group": "VOL::substrate",
   "kind": "dielectric",
   "material": "Si",
   "net": null,
   "shells": 1,
   "volume_um3": 100000.0
  }
 }
}
```
