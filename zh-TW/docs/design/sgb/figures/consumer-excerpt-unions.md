```{.json filename="semantic_geometry_route_b.sidecar.json（節錄）"}
{
 "insets": [
  {
   "band_count": 3,
   "bands": {
    "0": ["MA::pad::top::band=0"],
    "1": ["MA::pad::top::band=1"],
    "2": ["MA::pad::top::band=2"]
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
  "MA::pad::top::band=0": {
   "band": 0,
   "entities": ["pad"],
   "face_sides": null,
   "group": "MA::pad::top::band=0",
   "interface_type": "MA",
   "shared_face_group": null,
   "side": null
  },
  "SA::substrate|VACUUM_REGION::interface": {
   "band": null,
   "entities": ["VACUUM_REGION", "substrate"],
   "face_sides": null,
   "group": "SA::substrate|VACUUM_REGION::interface",
   "interface_type": "SA",
   "shared_face_group": null,
   "side": null
  }
 },
 "unions": {
  "interface_types": {
   "EXT": ["EXT::VACUUM_REGION::bottom", "EXT::VACUUM_REGION::sidewall", "EXT::VACUUM_REGION::top"],
   "MA": ["MA::ground_0000::sidewall", "MA::ground_0000::top", "MA::pad::sidewall", "MA::pad::top::band=0", "MA::pad::top::band=1", "MA::pad::top::band=2"],
   "MS": ["MS::ground_0000::bottom", "MS::pad::bottom"],
   "PORT": ["PORT::PORT_SHEET__P1__0000"],
   "SA": ["SA::substrate|VACUUM_REGION::interface", "SA::substrate|VACUUM_REGION::sidewall"]
  },
  "nets": {
   "G": ["MA::ground_0000::sidewall", "MA::ground_0000::top", "MS::ground_0000::bottom"],
   "P": ["MA::pad::sidewall", "MA::pad::top::band=0", "MA::pad::top::band=1", "MA::pad::top::band=2", "MS::pad::bottom"]
  }
 }
}
```
