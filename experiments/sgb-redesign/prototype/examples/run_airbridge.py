"""Run the airbridge variant (Route B, five slabs: posts and a deck over a gap)."""

from _common import run_case

from sgbproto.scenes import airbridge

if __name__ == "__main__":
    run_case(airbridge(), "airbridge_B", mesh_size=60.0, mesh_size_min=2.0)
