```{.json filename="semantic_geometry_route_b.sidecar.json（節錄）"}
{
 "insets": [
  {
   "band_count": 3,
   "bands": {
    "0": ["MA::pad__top::band=0"],
    "1": ["MA::pad__top::band=1"],
    "2": ["MA::pad__top::band=2"]
   },
   "interface": null,
   "margins_grid": [500, 1000],
   "margins_um": [0.5, 1.0],
   "request_id": "pad_top",
   "support": {
    "entity": "pad",
    "side": "top"
   },
   "support_area_um2": 400.0
  }
 ],
 "surface_epr_rows": {
  "MA::pad__top::band=0": {
   "band": 0,
   "entities": ["pad"],
   "face_sides": null,
   "group": "MA::pad__top::band=0",
   "interface_type": "MA",
   "shared_face_group": null,
   "side": null
  },
  "SA::substrate|VACUUM_REGION__interface": {
   "band": null,
   "entities": ["VACUUM_REGION", "substrate"],
   "face_sides": null,
   "group": "SA::substrate|VACUUM_REGION__interface",
   "interface_type": "SA",
   "shared_face_group": null,
   "side": null
  }
 },
 "unions": {
  "interface_types": {
   "EXT": ["EXT::VACUUM_REGION__bottom", "EXT::VACUUM_REGION__sidewall", "EXT::VACUUM_REGION__top"],
   "MA": ["MA::ground_0000__sidewall", "MA::ground_0000__top", "MA::pad__sidewall", "MA::pad__top::band=0", "MA::pad__top::band=1", "MA::pad__top::band=2"],
   "MS": ["MS::ground_0000__bottom", "MS::pad__bottom"],
   "PORT": ["PORT::PORT_SHEET__P1__0000"],
   "SA": ["SA::substrate|VACUUM_REGION__interface", "SA::substrate|VACUUM_REGION__sidewall"]
  },
  "nets": {
   "G": ["MA::ground_0000__sidewall", "MA::ground_0000__top", "MS::ground_0000__bottom"],
   "P": ["MA::pad__sidewall", "MA::pad__top::band=0", "MA::pad__top::band=1", "MA::pad__top::band=2", "MS::pad__bottom"]
  }
 }
}
```
