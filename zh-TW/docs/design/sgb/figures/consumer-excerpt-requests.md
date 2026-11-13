```{.json filename="semantic_geometry_route_b.sidecar.json（節錄）"}
{
 "band_family": {
  "pad_top": {
   "band_count": 3,
   "bands": {
    "0": ["MA::pad__top::band=0"],
    "1": ["MA::pad__top::band=1"],
    "2": ["MA::pad__top::band=2"]
   },
   "interface": null,
   "margins_grid": [500, 1000],
   "margins_um": [0.5, 1.0],
   "support": {
    "entity": "pad",
    "side": "top"
   },
   "support_area_um2": 400.0
  }
 },
 "cut_family": {
  "pad_top": {
   "levels_um": [0.2],
   "role": "inset_band"
  }
 },
 "inset_request": {
  "pad_top": {
   "families": ["pad_top"],
   "interface": null,
   "margins_um": [0.5, 1.0],
   "scope": "face",
   "skipped": [],
   "support": {
    "entity": "pad",
    "side": "top"
   }
  }
 },
 "surface_epr_row": {
  "MA::pad__top::band=0": {
   "band": 0,
   "banded_surface": "MA::pad__top",
   "entities": ["pad"],
   "face_sides": null,
   "group": "MA::pad__top::band=0",
   "interface_type": "MA",
   "shared_face_group": null,
   "side": null
  },
  "SA::substrate|VACUUM_REGION__interface": {
   "band": null,
   "banded_surface": null,
   "entities": ["VACUUM_REGION", "substrate"],
   "face_sides": null,
   "group": "SA::substrate|VACUUM_REGION__interface",
   "interface_type": "SA",
   "shared_face_group": null,
   "side": null
  }
 }
}
```
