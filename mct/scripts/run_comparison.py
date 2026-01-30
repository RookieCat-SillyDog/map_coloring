#!/usr/bin/env python
"""
Run comparison between CSP and BP algorithms on map coloring tasks.

Usage:
    python scripts/run_comparison.py [--map MAP_NAME] [--config CONFIG_PATH]
"""

import os
import argparse
import json
from datetime import datetime

from map_coloring.core import load_config, get_output_path, ExperimentConfig
from map_coloring.agents.classical_search import CSPConfig, CSPSolver
from map_coloring.data.maps import MAPS
from map_coloring.analysis.stats import (
    get_backtrack_depth_distribution,
    get_conflict_count_per_var,
    get_search_efficiency_metrics,
)


def run_csp_on_map(map_name: str, map_data: dict, csp_config: CSPConfig = None) -> dict:
    """Run CSP solver on a single map."""
    if csp_config is None:
        csp_config = CSPConfig()

    solver = CSPSolver(
        regions=map_data['regions'],
        neighbors=map_data['neighbors'],
        colors=map_data.get('colors', ['red', 'green', 'blue', 'yellow']),
        config=csp_config,
    )

    start_time = datetime.now()
    solution = solver.solve()
    elapsed = (datetime.now() - start_time).total_seconds()

    metrics = {
        'map': map_name,
        'algorithm': 'CSP',
        'solved': solution is not None,
        'elapsed_seconds': elapsed,
        'steps': solver.logger.summary.total_steps,
        'backtracks': solver.logger.summary.total_backtracks,
    }

    if solution:
        metrics.update({
            'solution': solution,
            'max_depth': solver.logger.summary.max_depth_reached,
        })
        efficiency = get_search_efficiency_metrics(solver.logger)
        metrics['efficiency'] = efficiency

    return metrics


def run_experiment(config: ExperimentConfig, map_filter: str = None) -> dict:
    """Run comparison experiment on all or selected maps."""
    results = {
        'run_id': config.generate_run_id(),
        'timestamp': datetime.now().isoformat(),
        'maps': [],
    }

    csp_config = CSPConfig(
        var_ordering_strategy=config.csp.variable_ordering,
        value_ordering_strategy=config.csp.value_ordering,
        use_forward_checking=config.csp.forward_checking,
        max_backtracks=config.csp.max_backtracks,
        rng_seed=config.csp.rng_seed,
    )

    for map_name, map_data in MAPS.items():
        if map_filter and map_filter != map_name:
            continue

        print(f"\n=== Running on {map_name} ===")

        if 'colors' not in map_data:
            map_data = {**map_data, 'colors': ['red', 'green', 'blue', 'yellow']}

        result = run_csp_on_map(map_name, map_data, csp_config)
        results['maps'].append(result)

        print(f"  Solved: {result['solved']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Backtracks: {result['backtracks']}")
        print(f"  Time: {result['elapsed_seconds']:.3f}s")

    return results


def save_results(results: dict, config: ExperimentConfig):
    """Save results to JSON file."""
    paths = get_output_path(results['run_id'], config=config)

    os.makedirs(paths['stats_dir'], exist_ok=True)
    os.makedirs(paths['plots_dir'], exist_ok=True)

    metrics_path = os.path.join(paths['stats_dir'], 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n=== Results saved ===")
    print(f"  Metrics: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Run CSP vs BP comparison')
    parser.add_argument('--map', '-m', type=str, default=None,
                        help='Run on specific map only')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results')

    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 50)
    print("Map Coloring - Algorithm Comparison")
    print("=" * 50)
    print(f"\nConfig:")
    print(f"  CSP variable ordering: {config.csp.variable_ordering}")
    print(f"  CSP value ordering: {config.csp.value_ordering}")
    print(f"  Forward checking: {config.csp.forward_checking}")

    results = run_experiment(config, args.map)

    if not args.no_save:
        save_results(results, config)

    print("\n=== Summary ===")
    solved = sum(1 for r in results['maps'] if r['solved'])
    print(f"  Maps tested: {len(results['maps'])}")
    print(f"  Solved: {solved}")
    print(f"  Failed: {len(results['maps']) - solved}")

    if solved > 0:
        avg_steps = sum(r['steps'] for r in results['maps'] if r['solved']) / solved
        avg_time = sum(r['elapsed_seconds'] for r in results['maps'] if r['solved']) / solved
        print(f"  Avg steps: {avg_steps:.1f}")
        print(f"  Avg time: {avg_time:.3f}s")


if __name__ == "__main__":
    main()
