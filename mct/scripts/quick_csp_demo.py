#!/usr/bin/env python
"""Quick CSP solver demo entry point."""

from map_coloring.agents.classical_search import CSPConfig, CSPSolver
from map_coloring.data.maps import MAPS


def main():
    """Run a quick CSP demo on the pentagon map."""
    map_name = "pentagon"
    map_config = MAPS[map_name]

    print("=" * 50)
    print("Quick CSP Solver Demo")
    print("=" * 50)

    # Use MRV + Forward Checking for efficiency
    config = CSPConfig(
        var_ordering_strategy="mrv",
        use_forward_checking=True,
    )

    solver = CSPSolver(
        regions=map_config["regions"],
        neighbors=map_config["neighbors"],
        colors=map_config["colors"],
        config=config,
    )

    solution = solver.solve()

    if solution:
        print("\n找到解：")
        for region, color in solution.items():
            print(f"  {region} -> {color}")
    else:
        print("\n未找到解")

    solver.logger.print_summary()


if __name__ == "__main__":
    main()
