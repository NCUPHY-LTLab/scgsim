"""Version-bound derivation identities and structured physical record vocabulary."""

GSIM_SHA = "8f5dc6c05255d003a9c6d8959537bcf8068379d3"

GSIM_HIGH_ORDER_DERIVATION_SHA = "00444a5fa343e2024ccf4f1fb6a3f50170166be4"

SGB_RUNTIME_AUTHORITY = "scgsim.geometry.compiler"

SGB_DERIVATION_BASE_SHA = "e74a343154c6b19b6ba32d6fb297e700cfe08ff2"

SGB_DERIVATION_IMPORTED_SHA = "f3fd898d6e4eaf31595c9aaca6a0658f0cb7f3b1"

STRUCTURED_INTERFACE_TYPES = {"MA", "MS", "SA", "MS_MA", "AA", "MM", "SS"}

_FACE_KIND_SEGMENTS = {"TOP": "top", "BOTTOM": "bottom", "SIDEWALL": "sidewall"}

STRUCTURED_FACE_KINDS = {
    **_FACE_KIND_SEGMENTS,
    "INTERFACE": "interface",
    "SHEET_CONTACT_CAP": "sheet_contact_cap",
}

STRUCTURED_SURFACE_FIELDS = frozenset(
    (
        "representation",
        "surface_id",
        "interface_type",
        "contact_kind",
        "face_kind",
        "owner_semantic_ids",
        "adjacent_solution_volume_ids",
        "conductor_component_id",
        "net_id",
        "equipotential_id",
        "source_provenance",
        "physical_attribute",
    )
)
