"""
CSP Solver - 可配置的约束满足问题求解器

模块结构：
- config.py: CSPConfig 配置结构
- logger.py: SearchLogger 日志记录器
- solver.py: 核心搜索循环
"""

from map_coloring.agents.classical_search.config import CSPConfig
from map_coloring.agents.classical_search.logger import (
    SearchLogger,
    SearchStepRecord,
    BacktrackEvent,
    VariableStats,
    EdgeStats,
    SearchRunSummary,
)
from map_coloring.agents.classical_search.solver import CSPSolver

__all__ = [
    'CSPConfig',
    'SearchLogger',
    'SearchStepRecord',
    'BacktrackEvent',
    'VariableStats',
    'EdgeStats',
    'SearchRunSummary',
    'CSPSolver',
]
