```{.json filename="groups_from_sidecar(sidecar)（節錄）"}
{
 "boundary_surfaces": {
  "MA::pad__top::band=0": {
   "adjacent_domains": [
    {
     "kind": "vacuum",
     "material": "vacuum",
     "name": "VACUUM_REGION"
    }
   ],
   "adjacent_volumes": ["vol:7346ec1a4ff1ccf9"],
   "area_um2": 39.0,
   "attributes": [7],
   "band": {
    "family": "pad_top",
    "index": 0
   },
   "dim": 2,
   "entities": ["pad"],
   "face_kind": "top",
   "group": "MA::pad__top::band=0",
   "interface_type": "MA",
   "net": "P",
   "rows": {
    "MA::pad__top::band=0": {
     "band": 0,
     "entities": ["pad"],
     "face_kind": "top",
     "face_sides": null,
     "family": "pad_top",
     "family_bands": {
      "0": ["MA::pad__top::band=0"],
      "1": ["MA::pad__top::band=1"],
      "2": ["MA::pad__top::band=2"]
     },
     "grid_um": 0.001,
     "group": "MA::pad__top::band=0",
     "interface_type": "MA",
     "margins_grid": [500, 1000],
     "margins_um": [0.5, 1.0],
     "net": "P",
     "shared_face_group": null,
     "side": null
    }
   },
   "source": "volume_interface",
   "surface_epr": true,
   "tags": [19]
  }
 },
 "pec_surfaces": {
  "MA::pad__top::band=0": {
   "attributes": [7],
   "dim": 2,
   "entities": ["pad"],
   "entity": "pad",
   "group": "MA::pad__top::band=0",
   "interface_type": "MA",
   "net": "P",
   "tags": [19]
  }
 },
 "port_surfaces": {
  "P1": {
   "adjacent_volumes": ["vol:1a3729bef5cef625", "vol:7346ec1a4ff1ccf9"],
   "attributes": [12],
   "dim": 2,
   "direction": [1.0, 0.0, 0.0],
   "faces": ["f:60d2e963bf0b88ab"],
   "group": "PORT::PORT_SHEET__P1__0000",
   "host_entity_ids": ["ground_0000", "pad"],
   "host_volumes": [],
   "parts": [
    {
     "attribute": 12,
     "faces": ["f:60d2e963bf0b88ab"],
     "membership": {
      "f:60d2e963bf0b88ab": {
       "PORT_SHEET__P1__0000": 1.0,
       "VACUUM_REGION": 1.0,
       "substrate": 1.0
      }
     },
     "name": "PORT::PORT_SHEET__P1__0000",
     "rows": [],
     "tags": [5]
    }
   ],
   "port_index": 1,
   "port_sheet_id": "PORT_SHEET__P1__0000",
   "source_layer": "9/0",
   "source_name": "P1",
   "tags": [5],
   "target_layer": "metal",
   "z_um": 0.0
  }
 },
 "volumes": {
  "substrate": {
   "attributes": [2],
   "dim": 3,
   "entities": ["substrate"],
   "group": "substrate",
   "kind": "dielectric",
   "material": "Si",
   "net": null,
   "tags": [1]
  }
 }
}
```

各鍵的項目數：`volumes` 2、`boundary_surfaces` 13、`pec_surfaces` 8、`port_surfaces` 1。
