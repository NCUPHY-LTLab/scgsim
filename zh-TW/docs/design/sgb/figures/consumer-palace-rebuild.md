```{.json filename="palace_style_groups(sidecar)（節錄）"}
{
 "boundary_surfaces": {
  "MA::pad__top::band=0": {
   "adjacent_solution_volume_ids": ["VACUUM_REGION"],
   "face_kind": "top",
   "interface_type": "MA",
   "net_id": "P",
   "owner_semantic_ids": ["pad"],
   "phys_group": 7
  }
 },
 "pec_surfaces": {
  "MA::pad__top::band=0": {
   "adjacent_solution_volume_ids": ["VACUUM_REGION"],
   "face_kind": "top",
   "interface_type": "MA",
   "net_id": "P",
   "owner_semantic_ids": ["pad"],
   "phys_group": 7
  }
 },
 "port_surfaces": {
  "PORT::PORT_SHEET__P1__0000": {
   "adjacent_solution_volume_ids": ["VACUUM_REGION", "substrate"],
   "face_kind": "port",
   "interface_type": "PORT",
   "net_id": null,
   "owner_semantic_ids": ["PORT_SHEET__P1__0000"],
   "phys_group": 12
  }
 },
 "volumes": {
  "substrate": {
   "material_kind": "dielectric",
   "owner_semantic_ids": ["substrate"],
   "phys_group": 2
  }
 }
}
```

各鍵的項目數：`volumes` 2、`boundary_surfaces` 14、`pec_surfaces` 8、`port_surfaces` 1。
