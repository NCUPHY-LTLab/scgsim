"""The English and Traditional Chinese labels of the geometry-build figures and tables."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")


TEXT = {
    "en": {
        "features": "features N",
        "seconds": "seconds",
        "megabytes": "megabytes",
        "derive": "derivation and validation",
        "slope_one": "slope 1 from N = 1,000",
        "memory": "peak RSS",
        "api": "Gmsh API",
        "direct": "direct writer",
        "occ_build": "OCC build",
        "synchronize": "synchronize",
        "groups": "addPhysicalGroup",
        "write_xao": "write XAO",
        "tail_one": "one process",
        "t_derive": "derivation and validation",
        "t_write": "write BREP and XAO",
        "t_reports": "reports",
        "t_sidecar": "sidecar",
        "t_load": "Gmsh load",
        "t_verify": "verify_loaded",
        "seconds_at_20000": "seconds at N = 20,000",
        "check_seconds": "seconds at N = 20,000",
        "structural": "structural check",
        "loaded": "check on the loaded model",
        "structural_total": "structural checks, total",
        "loaded_total": "verify_loaded, total",
        "faces": "faces F",
    },
    "zh-TW": {
        "features": "特徵數 N",
        "seconds": "秒",
        "megabytes": "MB",
        "derive": "推導與驗證",
        "slope_one": "以 N = 1,000 為起點的斜率 1",
        "memory": "peak RSS",
        "api": "Gmsh API",
        "direct": "直接寫出",
        "occ_build": "OCC 建構",
        "synchronize": "synchronize",
        "groups": "addPhysicalGroup",
        "write_xao": "寫出 XAO",
        "tail_one": "單一行程",
        "t_derive": "推導與驗證",
        "t_write": "寫出 BREP 與 XAO",
        "t_reports": "報告",
        "t_sidecar": "sidecar",
        "t_load": "Gmsh 載入",
        "t_verify": "verify_loaded",
        "seconds_at_20000": "N = 20,000 時的秒數",
        "check_seconds": "N = 20,000 時的秒數",
        "structural": "結構檢查",
        "loaded": "載入後模型的檢查",
        "structural_total": "結構檢查合計",
        "loaded_total": "verify_loaded 合計",
        "faces": "face 數 F",
    },
}

CHECKS = {  # name: (kind, English, Chinese); kind is "structural" or "loaded"
    "planarity": (
        "structural",
        "each arrangement edge, with the noding kernel",
        "各 arrangement 的 edge，用 noding kernel",
    ),
    "face_areas": (
        "loaded",
        "one Gmsh mass call per face",
        "每個 face 一次 Gmsh mass 呼叫",
    ),
    "shells": ("structural", "the faces of each shell", "每個 shell 的 face"),
    "counts_and_vertex_placement": (
        "loaded",
        "one Gmsh call per vertex",
        "每個 vertex 一次 Gmsh 呼叫",
    ),
    "boundaries": (
        "loaded",
        "one getBoundary call per edge, face and solid",
        "每個 edge、face、solid 一次 getBoundary",
    ),
    "volumes": ("loaded", "one mass integral per solid", "每個 solid 一次 mass 積分"),
    "physical_groups": (
        "loaded",
        "one group query per entity",
        "每個 entity 查詢一次 group",
    ),
    "groups": (
        "structural",
        "group membership of every face and volume",
        "每個 face 與 volume 的 group 歸屬",
    ),
    "edges_in_faces": (
        "structural",
        "the edges and vertical edges of every face",
        "每個 face 的 edge 與 vertical edge",
    ),
    "contact_check": (
        "structural",
        "contact between different Nets",
        "不同 Net 的接觸",
    ),
    "manifold_check": ("structural", "non-manifold pinches", "non-manifold pinch"),
    "atomic_ids": (
        "structural",
        "uniqueness of vertex, edge and face ids",
        "vertex、edge、face id 的唯一性",
    ),
    "grid_and_lengths": (
        "structural",
        "edge lengths and vertex grid positions",
        "edge 長度與 vertex 的 grid 位置",
    ),
    "face_sides": ("structural", "the two sides of every face", "每個 face 的兩側"),
    "areas": (
        "structural",
        "area conservation per arrangement",
        "各 arrangement 的面積守恆",
    ),
    "prism_areas": (
        "structural",
        "prism areas conserved on the cells",
        "prism 面積在 cell 上守恆",
    ),
    "cylinders": (
        "structural",
        "each curved wall is one cylinder",
        "每面 curved wall 是單一 cylinder",
    ),
    "bands": ("structural", "band areas are monotone", "band 面積單調"),
    "arcs": (
        "structural",
        "arc edges do not intersect and stay within the deviation",
        "arc edge 互不相交且在容許偏差內",
    ),
}

S9_CHECKS = (
    "grid_and_lengths",
    "atomic_ids",
    "planarity",
    "arcs",
    "bands",
    "areas",
    "prism_areas",
    "shells",
    "face_sides",
    "cylinders",
    "groups",
    "edges_in_faces",
)
