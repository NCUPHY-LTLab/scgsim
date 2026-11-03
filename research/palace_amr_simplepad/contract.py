"""Human-selected SimplePad Palace AMR study constants and case identities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PDK_REVISION = "c75a1d4d2ee07e86a3b5d9b491609e5a1a5b850c"
DEFAULT_STUDY_ROOT = Path(".artifacts/palace-amr-simplepad")
MPI_LAUNCHER = Path("/opt/openmpi/5.0.10/bin/mpirun")
MPI_PROCESSES = 30
OPENMP_THREADS = 2

PAD_SIDE_UM = 200.0
GAP_UM = 20.0
DIE_SIDE_UM = 1000.0
LOCATOR_WIDTH_UM = 2.0
SUBSTRATE_THICKNESS_UM = 500.0
SUBSTRATE_RELATIVE_PERMITTIVITY = 11.9
FILM_THICKNESS_UM = 0.1
VACUUM_PADDING_UM = (250.0, 250.0, 1000.0)

FEM_ORDER = 2
LINEAR_TOLERANCE = 1e-6
LINEAR_MAX_ITERATIONS = 200
AMR_MAX_ITERATIONS = 20
AMR_TOLERANCE = 0.01
AMR_THETA = 0.7
AMR_NC_LEVELS = 1

MESH_SIZE_POLICY = "gmsh_default"
MESH_ALGORITHM_3D = "Delaunay"
MESH_GEOMETRY_ORDER = 1
MESH_THREADS = 1
MESH_SURFACE_THREADS = 1


@dataclass(frozen=True)
class CaseDefinition:
    """Describe one route and its conformal or nonconformal refinement arm."""

    case_id: str
    route_id: str
    representation: Literal["A", "B"]
    approximation: Literal["thin_film", "route_b"]
    nonconformal: bool


CASES = (
    CaseDefinition("a-c", "route-a", "A", "thin_film", False),
    CaseDefinition("a-nc", "route-a", "A", "thin_film", True),
    CaseDefinition("b-c", "route-b", "B", "route_b", False),
    CaseDefinition("b-nc", "route-b", "B", "route_b", True),
)
