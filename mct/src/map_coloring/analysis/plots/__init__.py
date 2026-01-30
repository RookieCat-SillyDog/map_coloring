# -*- coding: utf-8 -*-
"""
可视化模块
"""

from map_coloring.analysis.plots.base import DEFAULT_COLORS, get_color, safe_normalize, compute_entropy
from map_coloring.analysis.plots.graph_coloring import build_graph, plot_coloring_state, create_coloring_legend
from map_coloring.analysis.plots.bp_plots import (
    visualize_decimation_process,
    visualize_bp_convergence,
    visualize_belief_evolution,
    print_belief_table,
    print_decimation_summary,
    create_bp_convergence_animation,
)
from map_coloring.analysis.plots.message_flow import (
    plot_message_flow_snapshot,
    plot_belief_state,
    create_message_flow_animation,
    visualize_message_propagation_steps,
    visualize_pure_bp_result,
    visualize_belief_evolution_pure_bp,
    create_pure_bp_animation,
)

__all__ = [
    # base
    'DEFAULT_COLORS', 'get_color', 'safe_normalize', 'compute_entropy',
    # graph_coloring
    'build_graph', 'plot_coloring_state', 'create_coloring_legend',
    # bp_plots
    'visualize_decimation_process', 'visualize_bp_convergence',
    'visualize_belief_evolution', 'print_belief_table', 'print_decimation_summary',
    'create_bp_convergence_animation',
    # message_flow
    'plot_message_flow_snapshot', 'plot_belief_state', 'create_message_flow_animation',
    'visualize_message_propagation_steps', 'visualize_pure_bp_result',
    'visualize_belief_evolution_pure_bp', 'create_pure_bp_animation',
]
