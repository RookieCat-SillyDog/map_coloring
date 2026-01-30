"""Statistical analysis and metrics."""
from map_coloring.analysis.stats.csp_search import (
    get_backtrack_depth_distribution,
    get_conflict_count_per_var,
    get_conflict_count_per_edge,
    get_variable_color_try_matrix,
    get_depth_profile,
    compute_var_difficulty_score,
    get_search_efficiency_metrics,
    get_action_distribution,
)

__all__ = [
    'get_backtrack_depth_distribution',
    'get_conflict_count_per_var',
    'get_conflict_count_per_edge',
    'get_variable_color_try_matrix',
    'get_depth_profile',
    'compute_var_difficulty_score',
    'get_search_efficiency_metrics',
    'get_action_distribution',
]
