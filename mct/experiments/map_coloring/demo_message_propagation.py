# -*- coding: utf-8 -*-
"""
BP 消息传播演示

展示 BP 的并行消息传播特性，对比 BP+Decimation 的串行决策。
用于说明 BP 与搜索算法的本质区别。

两种模式：
1. Pure BP：只运行 BP，不进行 Decimation，观察消息如何并行传播
2. BP+Decimation：观察每一步 Decimation 后，消息传播模式的变化
"""

from pathlib import Path
from typing import Optional

import numpy as np

from map_coloring.agents.bayesian import (
    BPDecimationConfig, run_decimation,
    PureBPConfig, run_pure_bp,
)
from map_coloring.data.maps import MAPS, get_edges, get_layout
from map_coloring.analysis.plots import (
    visualize_decimation_process,
    visualize_belief_evolution,
    print_belief_table,
    print_decimation_summary,
    visualize_pure_bp_result,
    visualize_belief_evolution_pure_bp,
    create_pure_bp_animation,
)

# 当前目录（用于保存输出）
CURRENT_DIR = Path(__file__).parent
OUTPUT_DIR = CURRENT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def demo_pure_bp(
    map_name: str = "gala",
    num_colors: int = 3,
    initial_evidence: Optional[dict[str, int]] = None,
    damping: float = 0.0,
    max_iter: int = 100,
    show_animation: bool = True,
    show_steps: bool = True,
    show_belief_evolution: bool = True,
    show_plots: bool = False,
):
    """
    Pure BP 演示：展示消息如何从 evidence 向全图并行传播。
    
    关键观察点：
    - 即使只固定一个节点，信息也会"波前式"扩散到整个图
    - 树状图上，传播次数与树的直径相关
    - 无 evidence 时，对称性导致所有节点保持均匀分布
    """
    print("=" * 60)
    print("Pure BP Demo: Observe Parallel Message Propagation")
    print("=" * 60)
    
    map_cfg = MAPS[map_name]
    nodes = map_cfg["regions"]
    edges = get_edges(map_name)
    layout = get_layout(map_name)
    
    # 默认固定一个节点以打破对称性
    if initial_evidence is None:
        initial_evidence = {nodes[0]: 0}
    
    config = PureBPConfig(
        max_iter=max_iter,
        damping=damping,
        tolerance=1e-6,
        update_schedule="synchronous",
        use_max_product=False,
    )
    
    result = run_pure_bp(
        map_name=map_name,
        nodes=nodes,
        edges=edges,
        num_colors=num_colors,
        config=config,
        initial_evidence=initial_evidence,
        verbose=True,
        trace_beliefs=True,
        trace_messages=True,
    )
    
    # 生成可视化
    if show_steps:
        steps_path = OUTPUT_DIR / f"pure_bp_{map_name}.png"
        visualize_pure_bp_result(
            result,
            save_path=str(steps_path),
            show=show_plots,
            max_iters=6,
            layout=layout,
        )
    
    if show_belief_evolution:
        evolution_path = OUTPUT_DIR / f"pure_bp_belief_evolution_{map_name}.png"
        visualize_belief_evolution_pure_bp(
            result,
            save_path=str(evolution_path),
            show=show_plots,
        )
    
    if show_animation:
        anim_path = OUTPUT_DIR / f"pure_bp_animation_{map_name}.gif"
        create_pure_bp_animation(
            result,
            save_path=str(anim_path),
            duration_sec=5,
            layout=layout,
        )
    
    return result


def demo_bp_decimation_with_messages(
    map_name: str = "gala",
    num_colors: int = 3,
    initial_evidence: Optional[dict[str, int]] = None,
    damping: float = 0.5,
    show_decimation: bool = True,
    show_beliefs: bool = True,
    show_plots: bool = False,
):
    """
    BP + Decimation 演示：展示串行决策如何改变消息传播场。
    """
    print("\n" + "=" * 60)
    print("BP + Decimation Demo: Sequential Decisions Change the Field")
    print("=" * 60)
    
    map_cfg = MAPS[map_name]
    nodes = map_cfg["regions"]
    edges = get_edges(map_name)
    layout = get_layout(map_name)
    
    config = BPDecimationConfig(
        max_iter=100,
        damping=damping,
        tolerance=1e-3,
        update_schedule="synchronous",
        threshold_margin=0.1,
        confidence_gap=0.2,
    )
    
    result = run_decimation(
        map_name=map_name,
        nodes=nodes,
        edges=edges,
        num_colors=num_colors,
        config=config,
        initial_evidence=initial_evidence,
        verbose=True,
        trace_bp=True,
    )
    
    print_decimation_summary(result)
    print_belief_table(result)
    
    if show_decimation:
        dec_path = OUTPUT_DIR / f"bp_decimation_{map_name}.png"
        visualize_decimation_process(
            result,
            save_path=str(dec_path),
            show=show_plots,
            layout=layout,
        )
    
    if show_beliefs:
        belief_path = OUTPUT_DIR / f"bp_belief_evolution_{map_name}.png"
        visualize_belief_evolution(
            result,
            save_path=str(belief_path),
            show=show_plots,
        )
    
    return result


def demo_comparison(
    map_name: str = "gala",
    num_colors: int = 3,
    show_plots: bool = False,
):
    """
    对比演示：同时运行 Pure BP 和 BP+Decimation，突出差异。
    """
    print("\n" + "=" * 70)
    print("COMPARISON: Pure BP (Parallel Inference) vs BP+Decimation (Sequential)")
    print("=" * 70)
    
    map_cfg = MAPS[map_name]
    nodes = map_cfg["regions"]
    
    initial_evidence = {nodes[0]: 0}
    
    print(f"\nMap: {map_name}, Nodes: {len(nodes)}, Colors: {num_colors}")
    print(f"Initial evidence: {initial_evidence}")
    
    # 1. Pure BP
    print("\n--- Running Pure BP ---")
    pure_result = demo_pure_bp(
        map_name=map_name,
        num_colors=num_colors,
        initial_evidence=initial_evidence,
        damping=0.0,
        max_iter=10,
        show_animation=False,
        show_steps=True,
        show_belief_evolution=True,
        show_plots=show_plots,
    )
    
    # 2. BP + Decimation
    print("\n--- Running BP + Decimation ---")
    dec_result = demo_bp_decimation_with_messages(
        map_name=map_name,
        num_colors=num_colors,
        initial_evidence=initial_evidence,
        damping=0.5,
        show_decimation=True,
        show_beliefs=True,
        show_plots=show_plots,
    )
    
    # 对比总结
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nPure BP:")
    print(f"  - Iterations: {pure_result['bp_iter']}")
    print(f"  - Converged: {pure_result['bp_converged']}")
    print(f"  - Final state: All nodes have soft probabilities (no hard decisions)")
    
    print(f"\nBP + Decimation:")
    print(f"  - Steps: {dec_result['steps']}")
    print(f"  - Solved: {dec_result['solved']}")
    print(f"  - Conflicts: {len(dec_result['conflicts'])}")
    print(f"  - Final state: All nodes have hard assignments")
    
    print("\nKey Difference:")
    print("  - Pure BP: PARALLEL message passing, preserves uncertainty")
    print("  - Decimation: SEQUENTIAL decisions, reduces problem size each step")


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    
    parser = argparse.ArgumentParser(description="BP Message Propagation Demo")
    parser.add_argument("--mode", type=str, default="comparison",
                       choices=["pure", "decimation", "comparison"],
                       help="Demo mode")
    parser.add_argument("--map", type=str, default="gala",
                       help="Map name")
    parser.add_argument("--colors", type=int, default=3,
                       help="Number of colors")
    parser.add_argument("--show", action="store_true",
                       help="Show plots interactively (default: save only)")
    
    args = parser.parse_args()
    
    show_plots = args.show
    
    if args.mode == "pure":
        demo_pure_bp(
            map_name=args.map, 
            num_colors=args.colors,
            show_steps=True,
            show_belief_evolution=True,
            show_animation=False,
        )
    elif args.mode == "decimation":
        demo_bp_decimation_with_messages(map_name=args.map, num_colors=args.colors)
    else:
        demo_comparison(map_name=args.map, num_colors=args.colors)
