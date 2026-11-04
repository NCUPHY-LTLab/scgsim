"""Run the pad-in-ground running example for Route A (sheet metal) and Route B (finite metal)."""

from _common import run_case

from sgbproto.scenes import pad_in_ground

if __name__ == "__main__":
    run_case(pad_in_ground("A"), "pad_in_ground_A", mesh_size=40.0)
    run_case(pad_in_ground("B"), "pad_in_ground_B", mesh_size=60.0, mesh_size_min=2.0)
