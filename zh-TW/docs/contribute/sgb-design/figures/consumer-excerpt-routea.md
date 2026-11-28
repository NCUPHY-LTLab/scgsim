```{.json filename="semantic_geometry_route_a.sidecar.json（節錄）"}
{
 "group": {
  "chip/left::band=0": {
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
   "interface_type": "SHEET",
   "net": "left",
   "tag": 10
  }
 },
 "model": {
  "route": "A",
  "route_policy": {
   "conductor_sheets": true,
   "conductor_volumes": "excluded"
  }
 },
 "port": {
  "port_sheet": {
   "adjacent_volumes": ["vol:b023b18d9f7c4ff3", "vol:b404a587b16d7174"],
   "group": "PORT::port_sheet",
   "host_entity_ids": ["chip/left", "chip/right"],
   "host_volumes": []
  }
 },
 "surface_epr_row": {
  "MA::chip/left__sheet::band=0": {
   "band": 0,
   "banded_surface": "MA::chip/left__sheet",
   "face_sides": {
    "f:5d1f74bc23d16ca3": "above"
   },
   "group": "chip/left::band=0",
   "interface_type": "MA",
   "shared_face_group": "chip/left::band=0",
   "side": "above"
  },
  "MS::chip/left__sheet::band=0": {
   "band": 0,
   "banded_surface": "MS::chip/left__sheet",
   "face_sides": {
    "f:5d1f74bc23d16ca3": "below"
   },
   "group": "chip/left::band=0",
   "interface_type": "MS",
   "shared_face_group": "chip/left::band=0",
   "side": "below"
  }
 }
}
```
