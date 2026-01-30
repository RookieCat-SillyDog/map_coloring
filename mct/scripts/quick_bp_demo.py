#!/usr/bin/env python
"""Quick BP + Decimation demo entry point."""

from map_coloring.agents.bayesian import BPDecimationConfig, run_decimation
from map_coloring.data.maps import MAPS, get_edges
from map_coloring.analysis.plots import print_decimation_summary, print_belief_table


def main():
    """Run a quick BP + Decimation demo on the gala map."""
    map_name = "gala"
    map_cfg = MAPS[map_name]
    nodes = map_cfg["regions"]
    edges = get_edges(map_name)
    num_colors = 3

    config = BPDecimationConfig(
        max_iter=100,
        tolerance=1e-3,
        damping=0.5,
        threshold_margin=0.1,
        confidence_gap=0.2,
    )

    print("=" * 50)
    print("Quick BP + Decimation Demo")
    print("=" * 50)

    result = run_decimation(
        map_name=map_name,
        nodes=nodes,
        edges=edges,
        num_colors=num_colors,
        config=config,
        verbose=True,
        trace_bp=False,
    )

    print_decimation_summary(result)
    print_belief_table(result)


if __name__ == "__main__":
    main()
