# -*- coding: utf-8 -*-
"""
BP 算法特有的可视化：belief 演化、收敛曲线等
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from map_coloring.analysis.plots.base import DEFAULT_COLORS, get_color, safe_normalize
from map_coloring.analysis.plots.graph_coloring import build_graph, plot_coloring_state, create_coloring_legend


def create_bp_convergence_animation(
    result: Dict[str, Any],
    target_step_index: int = 1,
    save_path: str = "bp_convergence.gif",
    duration_sec: int = 5,
):
    """
    创建 BP 在某一个 Decimation 步骤内的收敛过程动画 (GIF)。
    这能直观展示"消息是如何传播的"以及"同步/异步"的区别。
    
    Args:
        result: 运行结果
        target_step_index: 要可视化的 Decimation 步骤索引（默认 1，因为 Step 0 是初始，Step 1 通常变化最剧烈）
        save_path: 保存路径 (.gif)
        duration_sec: 动画持续时间
    """
    history = result['history']
    if target_step_index >= len(history):
        print(f"Step index {target_step_index} out of range (max {len(history)-1})")
        return

    record = history[target_step_index]
    
    # 检查是否有 trace 数据
    if 'bp_trace' not in record or not record['bp_trace']:
        print(f"No BP trace found for Step {target_step_index}. Did you run with trace_bp=True?")
        return

    bp_trace = record['bp_trace']
    # bp_trace 是一个 list[dict]，每个 dict 包含 {'iter': 1, 'beliefs': {...}}
    
    # 获取静态数据：图结构、布局
    nodes = result['nodes']
    edges = result['edges']
    num_colors = result['num_colors']
    
    # 构建图布局
    G, pos = build_graph(nodes, edges)
    
    # 准备画布
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 提取每一帧的 beliefs
    frames = []
    
    # 初始帧 (iter 0): 均匀分布
    uniform_beliefs = {n: np.ones(num_colors)/num_colors for n in nodes}
    frames.append({'iter': 0, 'beliefs': uniform_beliefs})
    
    # 后续帧
    for r in bp_trace:
        if 'beliefs' in r and r['beliefs']:
            frames.append(r)
            
    print(f"Generating animation with {len(frames)} frames for Step {target_step_index}...")

    def update(frame_idx):
        ax.clear()
        data = frames[frame_idx]
        current_iter = data['iter']
        current_beliefs = data.get('beliefs', {})
        
        # 将 list 形式的 belief 转回 numpy
        beliefs_np = {k: np.array(v) for k, v in current_beliefs.items()}
        
        # 使用当前步骤的 evidence (虽然 BP 过程中 evidence 不变，但作为背景显示)
        evidence = record['evidence']
        
        plot_coloring_state(
            ax=ax,
            G=G,
            pos=pos,
            assignment=evidence, # 这里的 evidence 是这一步已知的固定点
            beliefs=beliefs_np,
            num_colors=num_colors,
            title=f"BP Convergence - Step {target_step_index} - Iteration {current_iter}",
            show_beliefs=True,
            confidence_gap=result.get('config', {}).get('confidence_gap', 0.2), # Use config from result
            threshold_margin=result.get('config', {}).get('threshold_margin', 0.1)
        )
        
        # 添加图例
        # (简化处理，只加一次，或者每次让 plot_coloring_state 处理)
        # 这里因为 ax.clear()，需要每次都画一点简单的说明
        ax.text(0.5, -0.05, f"Observing message propagation...", 
                transform=ax.transAxes, ha='center', fontsize=10, color='gray')

    # 计算帧率
    fps = max(1, int(len(frames) / duration_sec))
    
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps)
    
    try:
        anim.save(save_path, writer=PillowWriter(fps=fps))
        print(f"Animation saved to: {save_path}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
        print("Ensure 'Pillow' is installed.")
    
    plt.close(fig)


def visualize_decimation_process(
    result: Dict[str, Any],
    save_path: Optional[str] = None,
    show: bool = True,
    max_steps: int = 12,
    layout: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[plt.Figure]:
    """
    可视化 Decimation 过程：每步的图着色状态
    
    Args:
        result: run_decimation 返回的结果字典
        save_path: 保存路径
        show: 是否显示
        max_steps: 最多显示的步数
        layout: 可选的节点布局，如果不提供则自动计算
    """
    history = result['history']
    nodes = result['nodes']
    edges = result['edges']
    num_colors = result['num_colors']
    map_name = result['map_name']
    cfg = result.get('config', {})
    threshold_margin = float(cfg.get('threshold_margin', 0.1))
    confidence_gap = float(cfg.get('confidence_gap', 0.0))
    initial_nodes = result.get('initial_nodes', [])
    
    G, pos = build_graph(nodes, edges, layout)
    
    # 限制显示步数
    display_history = history[:max_steps]
    n_steps = len(display_history)
    
    cols = min(n_steps, 4)
    rows = (n_steps + cols - 1) // cols
    
    fig = plt.figure(figsize=(5.5 * cols, 5.5 * rows))
    
    for idx, record in enumerate(display_history):
        ax = fig.add_subplot(rows, cols, idx + 1)
        
        # Use beliefs from the CURRENT step (priors before action)
        # Evidence includes the action taken in this step
        evidence = record['evidence']
        beliefs = {k: np.array(v) for k, v in record['beliefs'].items()}
        
        target = record.get('target')
        confident = record.get('confident', True)
        
        # 构建标题
        if target:
            target_node, target_color = target
            if confident:
                title = f"Step {record['step']}: Fix {target_node}={target_color}"
                title_color = 'green'
                highlight_style = 'confident'
            else:
                title = f"Step {record['step']}: Random {target_node}={target_color}"
                title_color = 'red'
                highlight_style = 'random'
            highlight_node = target_node
        else:
            if record['step'] == 0:
                title = "Initial State (BP Converged)"
                title_color = 'blue'
            else:
                title = f"Step {record['step']}: Complete"
                title_color = 'black'
            highlight_node = None
            highlight_style = 'fixed'
        
        plot_coloring_state(
            ax=ax,
            G=G,
            pos=pos,
            assignment=evidence,
            beliefs=beliefs,
            num_colors=num_colors,
            title=title,
            title_color=title_color,
            highlight_node=highlight_node,
            highlight_style=highlight_style,
            threshold_margin=threshold_margin,
            confidence_gap=confidence_gap,
            initial_nodes=initial_nodes,
        )
    
    # 图例
    legend_elements = create_coloring_legend(num_colors)
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=min(len(legend_elements), 6), fontsize=10, frameon=True)
    
    plt.suptitle(f"BP Decimation Process - {map_name} ({num_colors} Colors)",
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_bp_convergence(
    result: Dict[str, Any],
    save_path: Optional[str] = None,
    show: bool = True,
    max_steps: int = 6,
) -> Optional[plt.Figure]:
    """
    可视化 BP 收敛过程：残差曲线、熵变化等
    
    Args:
        result: run_decimation 返回的结果字典（需要 trace_bp=True）
        save_path: 保存路径
        show: 是否显示
        max_steps: 最多显示的步数
    """
    history = result['history']
    map_name = result['map_name']
    config = result.get('config', {})
    tolerance = config.get('tolerance', 1e-3)
    
    # 筛选有 bp_metrics 的步骤
    steps_with_trace = [h for h in history if h.get('bp_metrics')][:max_steps]
    
    if not steps_with_trace:
        print("No BP trace data available. Run with trace_bp=True.")
        return None
    
    n_steps = len(steps_with_trace)
    fig, axes = plt.subplots(n_steps, 2, figsize=(14, 3 * n_steps))
    if n_steps == 1:
        axes = axes.reshape(1, -1)
    
    for idx, record in enumerate(steps_with_trace):
        bp_metrics = record['bp_metrics']
        step_num = record['step']
        
        # 左图：残差收敛曲线
        ax_res = axes[idx, 0]
        residual_curve = bp_metrics.get('residual_curve', [])
        
        if residual_curve and any(r > 0 for r in residual_curve):
            iterations = list(range(1, len(residual_curve) + 1))
            # 过滤掉零值用于 log scale
            residual_plot = [max(r, 1e-12) for r in residual_curve]
            ax_res.semilogy(iterations, residual_plot, 'b-o', markersize=4, linewidth=1.5)
            ax_res.axhline(y=tolerance, color='r', linestyle='--', alpha=0.7, label=f'tol={tolerance}')
            ax_res.set_xlabel('Iteration')
            ax_res.set_ylabel('Residual (log)')
            ax_res.legend(loc='upper right', fontsize=9)
            ax_res.grid(True, alpha=0.3)
        else:
            ax_res.text(0.5, 0.5, 'No residual data', ha='center', va='center')
        
        ax_res.set_title(f'Step {step_num}: BP Convergence', fontweight='bold')
        
        # 右图：平均熵变化
        ax_ent = axes[idx, 1]
        entropy_curve = bp_metrics.get('mean_entropy_curve', [])
        
        if entropy_curve:
            iterations = list(range(1, len(entropy_curve) + 1))
            ax_ent.plot(iterations, entropy_curve, 'g-o', markersize=4, linewidth=1.5)
            ax_ent.set_xlabel('Iteration')
            ax_ent.set_ylabel('Mean Entropy')
            ax_ent.grid(True, alpha=0.3)
            
            # 标注最终值
            final_h = entropy_curve[-1]
            ax_ent.annotate(f'{final_h:.3f}', xy=(len(iterations), final_h),
                           xytext=(len(iterations) - 0.5, final_h + 0.1),
                           fontsize=10, fontweight='bold')
        else:
            ax_ent.text(0.5, 0.5, 'No entropy data', ha='center', va='center')
        
        ax_ent.set_title(f'Step {step_num}: Mean Entropy', fontweight='bold')
    
    plt.suptitle(f'BP Internal Iterations - {map_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_belief_evolution(
    result: Dict[str, Any],
    target_node: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    max_steps: int = 6,
) -> Optional[plt.Figure]:
    """
    可视化特定节点在 BP 迭代中的 belief 演化
    
    Args:
        result: run_decimation 返回的结果字典（需要 trace_bp=True）
        target_node: 目标节点（默认选择第一个未固定节点）
        save_path: 保存路径
        show: 是否显示
        max_steps: 最多显示的步数
    """
    history = result['history']
    num_colors = result['num_colors']
    map_name = result['map_name']
    
    # 筛选有 bp_trace 的步骤
    steps_with_trace = [h for h in history if h.get('bp_trace')][:max_steps]
    
    if not steps_with_trace:
        print("No BP trace data available. Run with trace_bp=True.")
        return None
    
    n_steps = len(steps_with_trace)
    cols = min(n_steps, 3)
    rows = (n_steps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_steps == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()
    
    for idx, record in enumerate(steps_with_trace):
        ax = axes[idx]
        bp_trace = record['bp_trace']
        unfixed = record.get('unfixed', [])
        step_num = record['step']
        target_info = record.get('target') # (node, color)
        
        # 选择目标节点逻辑：
        # 1. 如果用户指定了 target_node，尝试使用它（但要注意该节点必须还在图中，或者至少有belief数据）
        # 2. 如果是 Step > 0 且有 Decimation 目标，优先展示该目标节点（因为它是本步的主角）
        # 3. 否则展示第一个未固定节点
        node_to_plot = target_node
        
        # 如果用户未指定，或者指定的节点不在数据中（简单起见，只要没指定就自动选择）
        if node_to_plot is None:
            if target_info:
                node_to_plot = target_info[0]
            elif unfixed:
                node_to_plot = unfixed[0]
        
        if node_to_plot is None or not bp_trace:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'Step {step_num}')
            continue
        
        # 提取该节点的 belief 历史
        belief_history = []
        
        # bp_trace 中每条记录的 key 可能是 'iteration' 或 'iter'
        def get_iter(rec):
            return rec.get('iteration', rec.get('iter', 1))
        
        # 自动补全 Iteration 0 (均匀分布)，以便绘制从初始到第一次迭代的连线
        first_iter_beliefs = bp_trace[0].get('beliefs', {})
        if bp_trace and get_iter(bp_trace[0]) == 1 and node_to_plot in first_iter_beliefs:
            domain_size = len(first_iter_beliefs[node_to_plot])
            belief_history.append(np.ones(domain_size) / domain_size)

        valid_trace = False
        for rec in bp_trace:
            beliefs = rec.get('beliefs', {})
            if node_to_plot in beliefs:
                probs = safe_normalize(np.array(beliefs[node_to_plot]))
                belief_history.append(probs)
                valid_trace = True
        
        if not valid_trace:
            ax.text(0.5, 0.5, f'No belief data for {node_to_plot}', ha='center', va='center')
            ax.set_title(f'Step {step_num}: Node {node_to_plot}')
            continue
        
        # 绘制每个颜色的概率变化
        belief_array = np.array(belief_history)
        # x轴坐标：如果有补全0，则从0开始，否则从1开始(或依据trace数据的iter)
        start_iter = 0 if len(belief_history) > len(bp_trace) else get_iter(bp_trace[0])
        iterations = list(range(start_iter, start_iter + len(belief_history)))
        
        for c in range(min(num_colors, belief_array.shape[1])):
            ax.plot(iterations, belief_array[:, c],
                   color=get_color(c), marker='o', markersize=4,
                   linewidth=1.5, label=f'Color {c}')
        
        ax.axhline(y=1.0/num_colors, color='gray', linestyle=':', alpha=0.5, label='Uniform')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('P(color)')
        ax.set_ylim(-0.05, 1.05)
        # 为避免 Iter 0 和 Iter 1 重叠太近导致看不清，可以强制显示整数刻度
        if len(iterations) <= 5:
             ax.set_xticks(iterations)

        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Step {step_num}: Node "{node_to_plot}"', fontweight='bold')
    
    # 隐藏多余的子图
    for idx in range(n_steps, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Belief Evolution During BP - {map_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def print_belief_table(result: Dict[str, Any], precision: int = 2):
    """打印 Belief 变化表格（简化版）"""
    history = result['history']
    nodes = sorted(result['nodes'])
    num_colors = result['num_colors']
    
    print(f"\n{'='*60}")
    print("Belief Evolution Table")
    print(f"{'='*60}")
    
    # 计算列宽
    col_width = max(12, num_colors * 5 + 2)
    
    # 表头
    header = f"{'Step':<6}"
    for node in nodes:
        header += f"{node:^{col_width}}"
    print(header)
    print("-" * (6 + col_width * len(nodes)))
    
    for record in history:
        row = f"{record['step']:<6}"
        for node in nodes:
            if node in record['evidence']:
                prob_str = f"[{record['evidence'][node]}]"
            else:
                probs = record['beliefs'].get(node, [])
                if probs:
                    # 只显示 argmax 和概率
                    best = int(np.argmax(probs))
                    prob_str = f"{best}:{probs[best]:.2f}"
                else:
                    prob_str = "?"
            
            row += f"{prob_str:^{col_width}}"
        print(row)


def print_decimation_summary(result: Dict[str, Any]):
    """打印 Decimation 过程摘要"""
    history = result['history']
    
    print(f"\n{'='*70}")
    print("Decimation Summary")
    print(f"{'='*70}")
    
    total_random = sum(1 for h in history if not h.get('confident', True) and h.get('target'))
    total_confident = sum(1 for h in history if h.get('confident', True) and h.get('target'))
    
    print(f"Map: {result['map_name']}")
    print(f"Nodes: {len(result['nodes'])}, Edges: {len(result['edges'])}, Colors: {result['num_colors']}")
    print(f"Steps: {result['steps']}, Confident: {total_confident}, Random: {total_random}")
    print(f"Solved: {'Yes' if result['solved'] else 'No'}, Conflicts: {len(result['conflicts'])}")
    print(f"Time: {result.get('elapsed_seconds', 0):.3f}s")
    
    # BP 统计
    bp_iters = [h.get('bp_iter', 0) for h in history if h.get('bp_iter')]
    if bp_iters:
        print(f"BP iterations: avg={np.mean(bp_iters):.1f}, max={max(bp_iters)}")
