```{.json filename="semantic_geometry_route_b.sidecar.json（節錄）"}
{
 "carrier": {
  "ref/seg0": {
   "kind": "refinement",
   "request": "ref"
  }
 },
 "edge": {
  "e:3104024764366180": {
   "arc": null,
   "carriers": ["ref/seg0"],
   "entities": [],
   "kind": "refinement",
   "length_um": 10.0,
   "vertices": ["v:631126c5c6c687e9", "v:a5b0dda38b9136d7"],
   "written_as_free_edge": true
  }
 },
 "face": {
  "f:7e8c90cb3e7cf1ae": {
   "...": "...",
   "embedded_edges": ["e:3104024764366180"]
  }
 },
 "mesh_feature": {
  "mf:27de51bd5dbc2cca": {
   "edges": ["e:c5222f81df043e40", "e:5d7fe5923fb36d8a", "e:7e88aa41fe5fca70", "..."],
   "facts": {},
   "kind": "port_edge",
   "source": "PORT_SHEET__P1__0000"
  },
  "mf:69c04b90ee1d195f": {
   "edges": ["e:d8e8ce98d30b1b72", "e:635bebfe83d9ec65", "e:0e141478392a7c6d", "..."],
   "facts": {
    "thickness_um": 0.2
   },
   "kind": "film_sidewall",
   "source": "ground_0000"
  },
  "mf:6a1af6da3a537c60": {
   "edges": ["e:36da9569a3e69f89", "e:b1fa3f35d39449f4", "e:0beeb07b9aa96902", "..."],
   "facts": {
    "margins_um": [0.5, 1.0],
    "narrowest_band_um": 0.5
   },
   "kind": "band_boundary",
   "source": "pad_top"
  },
  "mf:f7a462c87bed4529": {
   "edges": ["e:3104024764366180"],
   "facts": {},
   "kind": "refinement_curve",
   "source": "ref"
  }
 },
 "port": {
  "PORT_SHEET__P1__0000": {
   "adjacent_volumes": ["vol:1a3729bef5cef625", "vol:7346ec1a4ff1ccf9"],
   "direction": [1.0, 0.0, 0.0],
   "entity": "PORT_SHEET__P1__0000",
   "faces": ["f:60d2e963bf0b88ab"],
   "group": "PORT::PORT_SHEET__P1__0000",
   "host_entity_ids": ["ground_0000", "pad"],
   "host_volumes": [],
   "port_index": 1,
   "source_layer": "9/0",
   "source_name": "P1",
   "target_layer": "metal",
   "z_um": 0.0
  }
 }
}
```
