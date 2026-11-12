```{.json filename="semantic_geometry_route_b.sidecar.json（節錄）"}
{
 "inset_request": {
  "sa": {
   "families": [
    "sa::substrate|VACUUM_REGION::z=0.0"
   ],
   "interface": null,
   "margins_um": [
    0.5,
    1.0
   ],
   "scope": "type",
   "skipped": [
    {
     "reason": "no_gap_edge",
     "support": {
      "first_entity": "substrate",
      "second_entity": "VACUUM_REGION",
      "z_um": -10.0
     }
    }
   ],
   "support": {
    "exclude_entity_ids": [],
    "face_kinds": null,
    "interface_type": "SA"
   }
  }
 },
 "band_family": {
  "sa::substrate|VACUUM_REGION::z=0.0": {
   "band_count": 3,
   "bands": {
    "0": [
     "SA::substrate|VACUUM_REGION__interface::band=0"
    ],
    "1": [
     "SA::substrate|VACUUM_REGION__interface::band=1"
    ],
    "2": [
     "SA::substrate|VACUUM_REGION__interface::band=2"
    ]
   },
   "interface": null,
   "margins_grid": [
    500,
    1000
   ],
   "margins_um": [
    0.5,
    1.0
   ],
   "support": {
    "first_entity": "substrate",
    "second_entity": "VACUUM_REGION",
    "z_um": 0.0
   },
   "support_area_um2": 1200.0
  }
 }
}
```
