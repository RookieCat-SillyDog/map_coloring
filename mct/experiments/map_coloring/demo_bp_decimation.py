# -*- coding: utf-8 -*-
"""
BP + Decimation 演示
展示置信传播与 decimation 策略的地图着色求解过程
"""

import os
from typing import Optional
from pathlib import Path

from map_coloring.agents.bayesian import BPDecimationConfig, run_decimation
from map_coloring.data.maps import MAPS, get_edges, get_layout
from map_coloring.analysis.plots import (
    visualize_decimation_process,
    visualize_bp_convergence,
    visualize_belief_evolution,
    print_belief_table,
    print_decimation_summary,
    create_bp_convergence_animation,
)
from map_coloring.core.config import load_config

# 当前目录（用于保存输出）
CURRENT_DIR = Path(__file__).parent
OUTPUT_DIR = CURRENT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 可视化开关
SHOW_PLOT = True


def demo_basic(
    map_name: str = "gala",
    num_colors: Optional[int] = None,
    verbose: bool = True,
    trace_bp: bool = True,
    show_plots: bool = True,
    config_path: Optional[str] = None,
    initial_assignment: Optional[dict[str, int]] = None
):
    """
    基本演示：运行 BP + Decimation 并可视化
    
    Args:
        map_name: 地图名称
        num_colors: 颜色数量（默认使用地图配置）
        verbose: 是否打印过程日志
        trace_bp: 是否记录 BP 内部轨迹（用于可视化）
        show_plots: 是否显示可视化图表
        config_path: 配置文件路径（默认使用 configs/default.yaml）
        initial_assignment: 初始固定的节点赋值，如 {"C2": 0} 表示固定 C2 为颜色 0
    """
    # 加载配置
    exp_config = load_config(config_path)
    
    # 从配置创建 BPDecimationConfig
    cfg = BPDecimationConfig(
        max_iter=exp_config.bp.max_iterations,
        tolerance=exp_config.bp.tolerance,
        threshold_margin=exp_config.bp.threshold_margin,
        confidence_gap=exp_config.bp.confidence_gap,
        damping=exp_config.bp.damping,
        same_color_penalty=exp_config.bp.same_color_penalty,
        diff_color_reward=exp_config.bp.diff_color_reward,
        update_schedule=exp_config.bp.update_schedule,
    )
    
    if verbose:
        print(f"Loaded config: max_iter={cfg.max_iter}, tolerance={cfg.tolerance}, "
              f"damping={cfg.damping}, threshold_margin={cfg.threshold_margin}, "
              f"confidence_gap={cfg.confidence_gap}")
    map_cfg = MAPS[map_name]
    nodes = map_cfg["regions"]
    edges = get_edges(map_name)
    colors = 4 or len(map_cfg["colors"])

    # 运行求解
    result = run_decimation(
        map_name=map_name,
        nodes=nodes,
        edges=edges,
        num_colors=colors,
        config=cfg,
        verbose=verbose,
        trace_bp=trace_bp,
        initial_evidence=initial_assignment,
    )

    # 打印结果摘要
    print_decimation_summary(result)
    
    # 打印 belief 表格
    print_belief_table(result)

    # 可视化
    if show_plots and SHOW_PLOT:
        # 获取地图布局
        layout = get_layout(map_name)
        
        # 1. Decimation 过程可视化
        visualize_decimation_process(
            result,
            save_path=str(OUTPUT_DIR / f"bp_decimation_{map_name}.png"),
            show=True,
            layout=layout,
        )
        
        # 2. BP 收敛曲线（需要 trace_bp=True）
        if trace_bp:
            visualize_bp_convergence(
                result,
                save_path=str(OUTPUT_DIR / f"bp_convergence_{map_name}.png"),
                show=True,
            )
            
            # 3. Belief 演化
            visualize_belief_evolution(
                result,
                save_path=str(OUTPUT_DIR / f"bp_belief_evolution_{map_name}.png"),
                show=True,
            )
            
            # 4. 生成动画（展示 BP 内部的消息传递过程 - Step 1）
            print("Generating BP propagation animation...")
            create_bp_convergence_animation(
                result,
                target_step_index=1,
                save_path=str(OUTPUT_DIR / f"bp_propagation_{map_name}.gif"),
                duration_sec=7
            )

    return result


if __name__ == "__main__":
    result = demo_basic()
