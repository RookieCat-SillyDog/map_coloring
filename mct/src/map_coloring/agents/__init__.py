"""Agents layer: algorithm implementations."""

# Classical search is always available
from map_coloring.agents.classical_search import (
    CSPConfig,
    CSPSolver,
)

__all__ = [
    # Classical
    "CSPConfig",
    "CSPSolver",
]

# Bayesian agents require pgm-toolkit (optional dependency)
try:
    from map_coloring.agents.bayesian import (
        BPDecimationConfig,
        run_decimation,
        build_markov_network,
        PureBPConfig,
        run_pure_bp,
    )
    __all__.extend([
        "BPDecimationConfig",
        "run_decimation",
        "build_markov_network",
        "PureBPConfig",
        "run_pure_bp",
    ])
except ImportError:
    # pgm-toolkit not installed, bayesian agents not available
    pass
