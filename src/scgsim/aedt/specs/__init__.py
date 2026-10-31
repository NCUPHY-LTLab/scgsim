"""AEDT request contracts: immutable specs, their JSON payload codecs, and validation.

Each module owns one family or concern:

- `common`: shared records (ports, materials, layer imports) and schema constants.
- `modeling`: physical-layer declarations and the source-to-effective Z map.
- `hfss`: HFSS requests that import GDS pieces, plus the shared run controls.
- `hfss_prepared`: HFSS requests built on a `PreparedPlanarGeometry`.
- `q3d`, `q2d`: Q3D and Q2D matrix-extraction requests.
- `epr`, `geometry`: EPR declarations and the prepared planar geometry carrier.
- `parse`: dispatch from a payload's `schema_version` to the right spec class.

Import from the owning module; this package re-exports nothing.
"""
