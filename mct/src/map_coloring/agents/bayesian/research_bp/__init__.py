"""
Research BP Module - 无侵入式 BP 轨迹与指标追踪

本模块在不修改原 pgm_toolkit 源码的前提下，通过复制并扩展 BP 引擎实现：
- BP 内部迭代轨迹 (t)
- 冲突度量（期望/实际）
- 收敛与振荡指标 (iter/residual/converged)

核心做法：
1. 在复制版 BP 的 loopy 循环中加入可选 trace/callback
2. 在 decimation 外层 step s 里读取并保存该 trace
3. 形成两层时间轴数据供打印/导出/可视化使用
"""

from map_coloring.agents.bayesian.research_bp.instrumented_belief_propagation import (
    InstrumentedBeliefPropagation,
    InstrumentedMaxProductBeliefPropagation,
    BPTrace,
    BPIterationRecord,
)

from map_coloring.agents.bayesian.research_bp.metrics import (
    compute_expected_violations,
    compute_map_conflicts,
    MetricsCalculator,
)

__all__ = [
    'InstrumentedBeliefPropagation',
    'InstrumentedMaxProductBeliefPropagation',
    'BPTrace',
    'BPIterationRecord',
    'compute_expected_violations',
    'compute_map_conflicts',
    'MetricsCalculator',
]
