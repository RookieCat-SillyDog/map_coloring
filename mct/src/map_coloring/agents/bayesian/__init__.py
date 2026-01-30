"""Bayesian inference agents (BP, decimation, etc.)."""

from map_coloring.agents.bayesian.research_bp import (
    InstrumentedBeliefPropagation,
    InstrumentedMaxProductBeliefPropagation,
    BPTrace,
    BPIterationRecord,
    compute_expected_violations,
    compute_map_conflicts,
    MetricsCalculator,
)
from map_coloring.agents.bayesian.decimation import BPDecimationConfig, run_decimation, build_markov_network
from map_coloring.agents.bayesian.pure_bp import PureBPConfig, run_pure_bp

__all__ = [
    "InstrumentedBeliefPropagation",
    "InstrumentedMaxProductBeliefPropagation",
    "BPTrace",
    "BPIterationRecord",
    "compute_expected_violations",
    "compute_map_conflicts",
    "MetricsCalculator",
    "BPDecimationConfig",
    "run_decimation",
    "build_markov_network",
    "PureBPConfig",
    "run_pure_bp",
]
