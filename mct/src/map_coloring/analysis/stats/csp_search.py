"""
模块 4：指标提取工具（离线分析函数）
从 SearchLogger 记录的日志中计算研究需要的统计量
"""

from typing import Optional
from collections import defaultdict

from map_coloring.agents.classical_search.logger import SearchLogger, SearchRunSummary


def get_backtrack_depth_distribution(logger: SearchLogger) -> list[int]:
    """
    获取回溯深度分布
    
    Args:
        logger: 搜索日志记录器
    
    Returns:
        回溯深度列表（每次回溯跨越的深度）
    """
    return [event.backtrack_depth for event in logger.backtrack_events]


def get_conflict_count_per_var(logger: SearchLogger) -> dict[str, int]:
    """
    获取每个变量的冲突次数
    
    Args:
        logger: 搜索日志记录器
    
    Returns:
        变量 -> 冲突次数 的字典
    """
    return {var: stats.reject_count for var, stats in logger.variable_stats.items()}


def get_conflict_count_per_edge(logger: SearchLogger) -> dict[tuple[str, str], int]:
    """
    获取每条边的冲突次数
    
    Args:
        logger: 搜索日志记录器
    
    Returns:
        边 (u, v) -> 冲突次数 的字典
    """
    edge_stats = logger.get_all_edge_stats()
    return {edge: stats.conflict_count for edge, stats in edge_stats.items()}


def get_variable_color_try_matrix(logger: SearchLogger) -> dict[str, dict[str, int]]:
    """
    获取变量-颜色尝试矩阵
    
    Args:
        logger: 搜索日志记录器
    
    Returns:
        变量 -> {颜色 -> 尝试次数} 的嵌套字典
    """
    return {var: dict(stats.color_try_counts) for var, stats in logger.variable_stats.items()}


def get_depth_profile(logger: SearchLogger) -> dict:
    """
    获取深度分析
    
    Args:
        logger: 搜索日志记录器
    
    Returns:
        包含深度统计的字典：
        - max_depth_reached: 达到的最大深度
        - histogram_of_depths: 每个深度的步数统计
        - avg_depth: 平均深度
    """
    depth_counts = defaultdict(int)
    total_depth = 0
    
    for step in logger.steps:
        depth_counts[step.depth] += 1
        total_depth += step.depth
    
    max_depth = logger.summary.max_depth_reached
    avg_depth = total_depth / len(logger.steps) if logger.steps else 0
    
    return {
        "max_depth_reached": max_depth,
        "histogram_of_depths": dict(depth_counts),
        "avg_depth": avg_depth,
        "depth_at_solution": max_depth if logger.summary.solved else None,
    }


def compute_var_difficulty_score(logger: SearchLogger) -> dict[str, float]:
    """
    计算变量难度分数
    
    基于以下因素加权计算：
    - reject_count: 拒绝次数（权重 0.4）
    - try_count / assign_count: 尝试/赋值比（权重 0.3）
    - 相关回溯事件数（权重 0.3）
    
    Args:
        logger: 搜索日志记录器
    
    Returns:
        变量 -> 难度分数 的字典
    """
    scores = {}
    
    # 计算每个变量相关的回溯次数
    var_backtrack_counts = defaultdict(int)
    for event in logger.backtrack_events:
        # 找到触发回溯的步骤
        for step in logger.steps:
            if step.step_id == event.step_id and step.var:
                var_backtrack_counts[step.var] += 1
                break
    
    # 归一化因子
    max_reject = max((s.reject_count for s in logger.variable_stats.values()), default=1) or 1
    max_backtrack = max(var_backtrack_counts.values(), default=1) or 1
    
    for var, stats in logger.variable_stats.items():
        # 拒绝分数
        reject_score = stats.reject_count / max_reject
        
        # 尝试/赋值比分数
        if stats.assign_count > 0:
            try_assign_ratio = stats.try_count / stats.assign_count
            ratio_score = min(try_assign_ratio / 10, 1.0)  # 归一化到 [0, 1]
        else:
            ratio_score = 1.0 if stats.try_count > 0 else 0.0
        
        # 回溯分数
        backtrack_score = var_backtrack_counts[var] / max_backtrack
        
        # 加权组合
        scores[var] = 0.4 * reject_score + 0.3 * ratio_score + 0.3 * backtrack_score
    
    return scores


def get_search_efficiency_metrics(logger: SearchLogger) -> dict:
    """
    获取搜索效率指标
    
    Args:
        logger: 搜索日志记录器
    
    Returns:
        效率指标字典
    """
    summary = logger.summary
    
    # 有效赋值率
    assign_efficiency = summary.total_assigns / summary.total_steps if summary.total_steps > 0 else 0
    
    # 回溯率
    backtrack_rate = summary.total_backtracks / summary.total_steps if summary.total_steps > 0 else 0
    
    # 平均回溯深度
    backtrack_depths = get_backtrack_depth_distribution(logger)
    avg_backtrack_depth = sum(backtrack_depths) / len(backtrack_depths) if backtrack_depths else 0
    
    return {
        "assign_efficiency": assign_efficiency,
        "backtrack_rate": backtrack_rate,
        "avg_backtrack_depth": avg_backtrack_depth,
        "total_steps": summary.total_steps,
        "total_assigns": summary.total_assigns,
        "total_rejects": summary.total_rejects,
        "total_backtracks": summary.total_backtracks,
    }


def get_action_distribution(logger: SearchLogger) -> dict[str, int]:
    """
    获取动作类型分布
    
    Args:
        logger: 搜索日志记录器
    
    Returns:
        动作类型 -> 次数 的字典
    """
    distribution = defaultdict(int)
    for step in logger.steps:
        distribution[step.action] += 1
    return dict(distribution)


def compare_runs(loggers: list[SearchLogger]) -> dict:
    """
    比较多次运行的结果
    
    Args:
        loggers: 日志记录器列表
    
    Returns:
        比较结果字典
    """
    results = []
    for i, logger in enumerate(loggers):
        results.append({
            "run_id": i,
            "solved": logger.summary.solved,
            "total_steps": logger.summary.total_steps,
            "total_backtracks": logger.summary.total_backtracks,
            "max_depth": logger.summary.max_depth_reached,
        })
    
    # 计算统计量
    solved_count = sum(1 for r in results if r["solved"])
    avg_steps = sum(r["total_steps"] for r in results) / len(results) if results else 0
    avg_backtracks = sum(r["total_backtracks"] for r in results) / len(results) if results else 0
    
    return {
        "runs": results,
        "summary": {
            "total_runs": len(results),
            "solved_count": solved_count,
            "solve_rate": solved_count / len(results) if results else 0,
            "avg_steps": avg_steps,
            "avg_backtracks": avg_backtracks,
        }
    }
