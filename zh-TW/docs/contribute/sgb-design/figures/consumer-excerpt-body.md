```{.json filename="semantic_geometry_route_b.sidecar.json（節錄）"}
{
 "binding": {
  "ids_by_tag": {
   "0": ["v:889c0afb53729cd6", "v:5ae72f37e0d2bca6", "..."],
   "1": ["e:f1a409262d686007", "e:717d8fb4f6b8f22d", "..."],
   "2": ["f:27ce1e55539dce2b", "f:2ea9880cc00e7a72", "..."],
   "3": ["vol:1a3729bef5cef625", "vol:7346ec1a4ff1ccf9"]
  },
  "rule": "planned id -> native tag: tag = 1 + position in ids_by_tag[dim] (TopExp::MapShapes pre-order of the written compound); never by geometry"
 },
 "carrier": {
  "ground_0000/polygon0/ring0/edge0": {
   "kind": "source",
   "source_curve": "ground_0000/polygon0/ring0/edge0"
  }
 },
 "edge": {
  "e:64197eb5e3183c98": {
   "arc": null,
   "carriers": ["ground_0000/polygon0/ring0/edge0"],
   "entities": [],
   "kind": "horizontal",
   "length_um": 30.0,
   "vertices": ["v:81719d79408a6890", "v:230e1a1c3d0a941d"]
  }
 },
 "entity": {
  "ground_0000": {
   "prisms": [
    {
     "feature_kind": "stack_film",
     "kind": "conductor",
     "layer": "metal",
     "material": "Al",
     "net": "G",
     "polygon": 0,
     "z_um": [0.0, 0.2]
    }
   ]
  }
 },
 "face": {
  "f:179b3dd8d7bb9c70": {
   "area_um2": 6.0,
   "band_families": [],
   "bands": [],
   "curved": false,
   "entities": ["ground_0000", "substrate"],
   "entity_coverage": {
    "ground_0000": 1.0,
    "substrate": 1.0
   },
   "face_kind": "sidewall",
   "group": "MA::ground_0000__sidewall",
   "kind": "wall",
   "loops": [
    {
     "edges": ["ve:5308f97729292679", "e:64197eb5e3183c98", "ve:d70909d0a0d61f80", "e:a527a49b4b34f917"],
     "reversed": [false, false, true, true]
    }
   ],
   "negative_volume": "vol:739993ecf3919fb3",
   "other_cut_families": [],
   "outer_boundary": null,
   "positive_volume": "vol:7346ec1a4ff1ccf9",
   "z_um": [0.0, 0.2]
  }
 },
 "source_curve": {
  "ground_0000/polygon0/ring0/edge0": {
   "curve": {
    "end_um": [0.0, 0.0],
    "kind": "line",
    "start_um": [0.0, 30.0]
   },
   "edge": 0,
   "entity": "ground_0000",
   "length_um": 30.0,
   "polygon": 0,
   "ring": 0
  }
 },
 "vertex": {
  "v:230e1a1c3d0a941d": {
   "position_um": [0.0, 30.0, 0.2]
  },
  "v:81719d79408a6890": {
   "position_um": [0.0, 0.0, 0.2]
  }
 },
 "volume": {
  "vol:1a3729bef5cef625": {
   "aedt_name": "substrate",
   "embedded_faces": [],
   "emitted": true,
   "entity": "substrate",
   "faces": ["f:27ce1e55539dce2b", "f:2ea9880cc00e7a72", "f:487ca41cd08e5769", "f:49c3dfd55c57dddd", "..."],
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
