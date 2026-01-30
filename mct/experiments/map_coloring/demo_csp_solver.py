"""
CSP Solver 演示
展示可配置求解器的使用方法
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from map_coloring.agents.classical_search import CSPConfig, CSPSolver
from map_coloring.analysis.stats import (
    get_backtrack_depth_distribution,
    get_conflict_count_per_var,
    get_conflict_count_per_edge,
    get_variable_color_try_matrix,
    get_depth_profile,
    compute_var_difficulty_score,
    get_search_efficiency_metrics,
    get_action_distribution,
)
from map_coloring.data.maps import MAPS

CURRENT_DIR = Path(__file__).parent
OUTPUT_DIR = CURRENT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 可视化配置
SHOW_PLOT = True
COLOR_MAP = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']


def demo_basic_solve():
    """基本求解演示"""
    print("=" * 60)
    print("演示 1：基本求解（默认配置）")
    print("=" * 60)
    
    map_config = MAPS["pentagon"]
    
    solver = CSPSolver(
        regions=map_config["regions"],
        neighbors=map_config["neighbors"],
        colors=map_config["colors"],
    )
    
    solution = solver.solve()
    
    if solution:
        print("\n找到解：")
        for region, color in solution.items():
            print(f"  {region} -> {color}")
    
    solver.logger.print_summary()
    solver.logger.print_steps(20)


def demo_with_forward_checking():
    """Forward Checking 演示"""
    print("\n" + "=" * 60)
    print("演示 2：启用 Forward Checking")
    print("=" * 60)
    
    map_config = MAPS["pentagon"]
    
    config = CSPConfig(
        use_forward_checking=True,
        var_ordering_strategy="mrv",
        value_ordering_strategy="random",
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
    
    solver.logger.print_summary()


def demo_analysis():
    """分析指标演示"""
    print("\n" + "=" * 60)
    print("演示 3：搜索分析指标")
    print("=" * 60)
    
    map_config = MAPS["pentagon"]
    
    solver = CSPSolver(
        regions=map_config["regions"],
        neighbors=map_config["neighbors"],
        colors=map_config["colors"],
    )
    
    solver.solve()
    logger = solver.logger
    
    # 回溯深度分布
    backtrack_depths = get_backtrack_depth_distribution(logger)
    print(f"\n回溯深度分布: {backtrack_depths}")
    
    # 每个变量的冲突次数
    conflict_per_var = get_conflict_count_per_var(logger)
    print(f"\n每个变量的冲突次数:")
    for var, count in conflict_per_var.items():
        if count > 0:
            print(f"  {var}: {count}")
    
    # 深度分析
    depth_profile = get_depth_profile(logger)
    print(f"\n深度分析:")
    print(f"  最大深度: {depth_profile['max_depth_reached']}")
    print(f"  平均深度: {depth_profile['avg_depth']:.2f}")
    
    # 动作分布
    action_dist = get_action_distribution(logger)
    print(f"\n动作分布: {action_dist}")


def demo_compare_strategies():
    """策略比较演示"""
    print("\n" + "=" * 60)
    print("演示 4：不同策略比较")
    print("=" * 60)
    
    map_config = MAPS["australia"]
    
    strategies = [
        ("静态顺序", CSPConfig(var_ordering_strategy="static", value_ordering_strategy="static")),
        ("MRV", CSPConfig(var_ordering_strategy="mrv", use_forward_checking=True)),
        ("MRV + LCV", CSPConfig(var_ordering_strategy="mrv", value_ordering_strategy="least_constraining", use_forward_checking=True)),
        ("随机", CSPConfig(var_ordering_strategy="random", value_ordering_strategy="random", rng_seed=42)),
    ]
    
    print(f"\n{'策略':<15} {'步数':<8} {'赋值':<8} {'回溯':<8} {'解'}")
    print("-" * 60)
    
    for name, config in strategies:
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
            config=config,
        )
        
        solution = solver.solve()
        summary = solver.logger.summary
        
        solved_str = "✓" if summary.solved else "✗"
        print(f"{name:<15} {summary.total_steps:<8} {summary.total_assigns:<8} {summary.total_backtracks:<8} {solved_str}")


if __name__ == "__main__":
    demo_basic_solve()
    demo_with_forward_checking()
    demo_analysis()
    demo_compare_strategies()
