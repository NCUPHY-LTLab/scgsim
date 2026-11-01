"""Source identities recorded in every Palace mesh artifact.

Owns the revision hashes and the geometry-authority name that the mesh manifest,
the route provenance file and the handoff metadata all quote. It does not decide
what is meshed; changing a value here changes recorded provenance only.

SGB is the semantic geometry builder, now in-tree at ``scgsim.geometry.compiler``.
"""

GSIM_SHA = "8f5dc6c05255d003a9c6d8959537bcf8068379d3"
GSIM_HIGH_ORDER_DERIVATION_SHA = "00444a5fa343e2024ccf4f1fb6a3f50170166be4"
SGB_RUNTIME_AUTHORITY = "scgsim.geometry.compiler"
# TODO(human): confirm what the two SGB derivation revisions identify
# (base revision and imported development revision of the SGB lineage).
SGB_DERIVATION_BASE_SHA = "e74a343154c6b19b6ba32d6fb297e700cfe08ff2"
SGB_DERIVATION_IMPORTED_SHA = "f3fd898d6e4eaf31595c9aaca6a0658f0cb7f3b1"
