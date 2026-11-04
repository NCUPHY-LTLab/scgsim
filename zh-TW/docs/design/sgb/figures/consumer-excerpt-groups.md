```{.json filename="semantic_geometry_route_b.sidecar.json（節錄）"}
{
 "physical_groups": {
  "MA::pad::top::band=0": {
   "adjacent_domains": [
    {
     "kind": "vacuum",
     "material": "vacuum",
     "name": "VACUUM_REGION"
    }
   ],
   "adjacent_volumes": ["vol:7346ec1a4ff1ccf9"],
   "area_um2": 39.0,
   "band": {
    "family": "pad_top",
    "index": 0
   },
   "dim": 2,
   "entities": ["pad"],
   "face_kind": "top",
   "ids": ["f:32215dfca2e5b36e"],
   "interface_type": "MA",
   "native_tags": [19],
   "net": "P",
   "tag": 7
  },
  "MS::pad::bottom": {
   "adjacent_domains": [
    {
     "kind": "dielectric",
     "material": "Si",
     "name": "substrate"
    }
   ],
   "adjacent_volumes": ["vol:1a3729bef5cef625"],
   "area_um2": 400.0,
   "band": null,
   "dim": 2,
   "entities": ["pad"],
   "face_kind": "bottom",
   "ids": ["f:120f732fa4504c88"],
   "interface_type": "MS",
   "native_tags": [1],
   "net": "P",
   "tag": 11
  },
  "PORT::PORT_SHEET__P1__0000": {
   "adjacent_domains": [
    {
     "kind": "vacuum",
     "material": "vacuum",
     "name": "VACUUM_REGION"
    },
    {
     "kind": "dielectric",
     "material": "Si",
     "name": "substrate"
    }
   ],
   "adjacent_volumes": ["vol:1a3729bef5cef625", "vol:7346ec1a4ff1ccf9"],
   "area_um2": 40.0,
   "band": null,
   "dim": 2,
   "entities": ["PORT_SHEET__P1__0000"],
   "face_kind": "port",
   "ids": ["f:2d1bc77f577fd23b"],
   "interface_type": "PORT",
   "native_tags": [3],
   "net": null,
   "tag": 12
  },
  "substrate": {
   "dim": 3,
   "entities": ["substrate"],
   "ids": ["vol:1a3729bef5cef625"],
   "interface_type": null,
   "kind": "dielectric",
   "material": "Si",
   "native_tags": [1],
   "net": null,
   "tag": 2
  }
 }
}
```
