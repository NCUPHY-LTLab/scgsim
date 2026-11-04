```{.json filename="semantic_geometry_route_a.sidecar.json（節錄）"}
{
 "embedded_faces": [],
 "physical_groups": {
  "SHEET::chip/left::sheet::band=0": {
   "adjacent_domains": [
    {
     "kind": "dielectric",
     "material": "Si",
     "name": "substrate"
    },
    {
     "kind": "vacuum",
     "material": "air",
     "name": "upper_air"
    }
   ],
   "adjacent_volumes": ["vol:b023b18d9f7c4ff3", "vol:b404a587b16d7174"],
   "area_um2": 7.0,
   "band": {
    "family": "band_left",
    "index": 0
   },
   "dim": 2,
   "entities": ["chip/left"],
   "face_kind": "sheet",
   "ids": ["f:c6a511988c40c746"],
   "interface_type": "SHEET",
   "native_tags": [7],
   "net": "left",
   "tag": 10
  }
 },
 "ports": [
  {
   "adjacent_volumes": ["vol:b023b18d9f7c4ff3", "vol:b404a587b16d7174"],
   "group": "PORT::port_sheet",
   "host_entity_ids": ["chip/left", "chip/right"],
   "host_volumes": []
  }
 ],
 "route_policy": {
  "conductor_sheets": true,
  "conductor_volumes": "excluded"
 },
 "surface_epr_rows": {
  "MA::chip/left::sheet::band=0": {
   "band": 0,
   "entities": ["chip/left"],
   "group": "SHEET::chip/left::sheet::band=0",
   "interface_type": "MA",
   "shared_face_group": "SHEET::chip/left::sheet::band=0",
   "side": "above"
  },
  "MS::chip/left::sheet::band=0": {
   "band": 0,
   "entities": ["chip/left"],
   "group": "SHEET::chip/left::sheet::band=0",
   "interface_type": "MS",
   "shared_face_group": "SHEET::chip/left::sheet::band=0",
   "side": "below"
  }
 },
 "unions": {
  "interface_types": {
   "AA": ["AA::lower_air|upper_air::interface"],
   "EXT": ["EXT::lower_air::bottom", "EXT::lower_air::sidewall", "EXT::substrate::bottom", "EXT::upper_air::sidewall", "EXT::upper_air::top"],
   "MA": ["SHEET::chip/left::sheet::band=0", "SHEET::chip/left::sheet::band=1", "SHEET::chip/left::sheet::band=2", "SHEET::chip/right::sheet"],
   "MS": ["SHEET::chip/left::sheet::band=0", "SHEET::chip/left::sheet::band=1", "SHEET::chip/left::sheet::band=2", "SHEET::chip/right::sheet"],
   "PORT": ["PORT::port_sheet"],
   "SA": ["SA::substrate|lower_air::sidewall", "SA::substrate|upper_air::interface"],
   "SHEET": ["SHEET::chip/left::sheet::band=0", "SHEET::chip/left::sheet::band=1", "SHEET::chip/left::sheet::band=2", "SHEET::chip/right::sheet"]
  },
  "nets": {
   "left": ["SHEET::chip/left::sheet::band=0", "SHEET::chip/left::sheet::band=1", "SHEET::chip/left::sheet::band=2"],
   "right": ["SHEET::chip/right::sheet"]
  }
 }
}
```
