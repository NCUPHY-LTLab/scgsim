```{.json filename="semantic_geometry_route_b.sidecar.json（節錄）"}
{
 "embedded_faces": [],
 "mesh_features": [
  {
   "edges": ["e:36da9569a3e69f89", "e:b1fa3f35d39449f4", "e:0beeb07b9aa96902", "..."],
   "facts": {
    "margins_um": [0.5, 1.0],
    "narrowest_band_um": 0.5
   },
   "kind": "band_boundary",
   "native_tags": [54, 55, 56, "..."],
   "source": "pad_top"
  },
  {
   "edges": ["e:d8e8ce98d30b1b72", "e:635bebfe83d9ec65", "e:0e141478392a7c6d", "..."],
   "facts": {
    "thickness_um": 0.2
   },
   "kind": "film_sidewall",
   "native_tags": [5, 6, 7, "..."],
   "source": "ground_0000"
  },
  {
   "edges": ["e:1090b0e21ca660dd", "e:41d0282b4480bda6", "e:d958f2000e51b6b5", "..."],
   "facts": {
    "thickness_um": 0.2
   },
   "kind": "film_sidewall",
   "native_tags": [11, 12, 13, "..."],
   "source": "pad"
  },
  {
   "edges": ["e:f1a409262d686007", "e:717d8fb4f6b8f22d", "e:ae5887e8b777374e", "..."],
   "facts": {
    "thickness_um": 10.0
   },
   "kind": "film_sidewall",
   "native_tags": [1, 2, 3, "..."],
   "source": "substrate"
  },
  {
   "edges": ["e:c5222f81df043e40", "e:5d7fe5923fb36d8a", "e:7e88aa41fe5fca70", "..."],
   "facts": {},
   "kind": "port_edge",
   "native_tags": [10, 16, 17, "..."],
   "source": "PORT_SHEET__P1__0000"
  },
  {
   "edges": ["e:3104024764366180"],
   "facts": {},
   "kind": "refinement_curve",
   "native_tags": [86],
   "source": "ref"
  }
 ],
 "ports": [
  {
   "adjacent_volumes": ["vol:1a3729bef5cef625", "vol:7346ec1a4ff1ccf9"],
   "direction": [1.0, 0.0, 0.0],
   "faces": ["f:60d2e963bf0b88ab"],
   "group": "PORT::PORT_SHEET__P1__0000",
   "host_entity_ids": ["ground_0000", "pad"],
   "host_volumes": [],
   "port_index": 1,
   "port_sheet_id": "PORT_SHEET__P1__0000",
   "source_layer": "9/0",
   "source_name": "P1",
   "target_layer": "metal",
   "z_um": 0.0
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
