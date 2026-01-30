# -*- coding: utf-8 -*-
"""
BP + Decimation 演示
展示置信传播与 decimation 策略的地图着色求解
"""

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pgm_toolkit.models.markovnet import MarkovNetwork
from agents.bayesian.research_bp import (
    InstrumentedMaxProductBeliefPropagation,
    compute_map_conflicts,
)
from tasks.experiment1_map_coloring.maps import MAPS, get_map, get_edges, get_color_domain

# 当前目录（用于保存输出）
current_dir = os.path.dirname(os.path.abspath(__file__))

# 可视化配置
SHOW_PLOT = True
COLOR_MAP = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']


@dataclass
class BPConfig:
    """BP 算法配置"""
    max_iter: int = 100
    tolerance: float = 1e-3
    damping: float = 0.0
    update_schedule: str = "synchronous"
    rng_seed: int = 0
    same_color_penalty: float = 1e-6
    diff_color_reward: float = 1.0


def _safe_normalize(probs: np.ndarray) -> np.ndarray:
    """安全归一化概率分布"""
    probs = np.asarray(probs, dtype=float)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(probs, 1.0 / len(probs), dtype=float)
    return probs / total


def _entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    """计算熵"""
    p = _safe_normalize(probs)
    return float(-np.sum(p * np.log(p + eps)))


def _summarize_bp_trace(bp_trace) -> dict:
    """汇总 BP 追踪信息"""
    if bp_trace is None or not getattr(bp_trace, "records", None):
        return {
            "n_iter": 0,
            "residual_curve": [],
            "mean_entropy_curve": [],
            "total_l1_curve": [],
            "argmax_flips": {},
            "total_flips": 0,
            "nodes_with_flips": 0,
            "max_flips_per_node": 0,
        }

    records = bp_trace.records
    residual_curve = [float(r.residual) for r in records]
    mean_entropy_curve = []
    total_l1_curve = []
    argmax_flips = {}
    prev_beliefs = None
    prev_argmax = {}

    for r in records:
        snapshot = r.beliefs or {}

        if snapshot:
            entropies = [_entropy(v) for v in snapshot.values()]
            mean_entropy_curve.append(
                float(np.mean(entropies)) if entropies else float("nan")
            )
        else:
            mean_entropy_curve.append(float("nan"))

        if prev_beliefs is not None and snapshot:
            common_vars = set(prev_beliefs.keys()) & set(snapshot.keys())
            l1_total = 0.0
            for var in common_vars:
                a = _safe_normalize(prev_beliefs[var])
                b = _safe_normalize(snapshot[var])
                l1_total += float(np.sum(np.abs(a - b)))
            total_l1_curve.append(l1_total)

        if snapshot:
            for var, probs in snapshot.items():
                amax = int(np.argmax(probs))
                if var in prev_argmax and prev_argmax[var] != amax:
                    argmax_flips[var] = argmax_flips.get(var, 0) + 1
                prev_argmax[var] = amax

        prev_beliefs = snapshot

    total_flips = int(sum(argmax_flips.values()))
    nodes_with_flips = int(sum(1 for v in argmax_flips.values() if v > 0))
    max_flips_per_node = int(max(argmax_flips.values())) if argmax_flips else 0

    return {
        "n_iter": len(records),
        "residual_curve": residual_curve,
        "mean_entropy_curve": mean_entropy_curve,
        "total_l1_curve": total_l1_curve,
        "argmax_flips": argmax_flips,
        "total_flips": total_flips,
        "nodes_with_flips": nodes_with_flips,
        "max_flips_per_node": max_flips_per_node,
    }


def build_markov_network(
    map_name: str,
    num_colors: Optional[int] = None,
    config: Optional[BPConfig] = None,
) -> MarkovNetwork:
    """
    构建马尔可夫网络
    
    Args:
        map_name: 地图名称
        num_colors: 颜色数量（默认使用地图配置中的颜色数）
        config: BP 配置
    
    Returns:
        MarkovNetwork 实例
    """
    if config is None:
        config = BPConfig()
    
    map_config = get_map(map_name)
    nodes = map_config["regions"]
    edges = get_edges(map_name)
    
    if num_colors is None:
        num_colors = len(map_config["colors"])
    
    color_domain = list(range(num_colors))
    
    mn = MarkovNetwork(name=f"{map_name}_coloring")
    for node in nodes:
        mn.add_node(node, domain=color_domain)
    
    for u, v in edges:
        table = np.full((num_colors, num_colors), config.diff_color_reward, dtype=float)
        for c in range(num_colors):
            table[c, c] = config.same_color_penalty
        mn.add_potential([u, v], table)
    
    return mn


def run_decimation(
    map_name: str,
    num_colors: Optional[int] = None,
    config: Optional[BPConfig] = None,
    verbose: bool = True,
    show_bp_trace: bool = True,
) -> dict:
    """
    运行 Decimation 求解
    
    Args:
        map_name: 地图名称
        num_colors: 颜色数量
        config: BP 配置
        verbose: 是否打印详细信息
        show_bp_trace: 是否显示 BP 追踪指标
    
    Returns:
        包含求解历史和结果的字典
    """
    if config is None:
        config = BPConfig()
    
    map_config = get_map(map_name)
    nodes = map_config["regions"]
    edges = get_edges(map_name)
    
    if num_colors is None:
        num_colors = len(map_config["colors"])
    
    color_domain = list(range(num_colors))
    threshold = 1.0 / num_colors + 0.1
    
    # 构建网络
    mn = build_markov_network(map_name, num_colors, config)
    
    # 创建推理引擎
    engine = InstrumentedMaxProductBeliefPropagation(
        max_iter=config.max_iter,
        tol=config.tolerance,
        damping=config.damping,
        update_schedule=config.update_schedule,
        rng_seed=config.rng_seed,
    )
    mn.set_inference_engine(engine)
    
    if verbose:
        print("=" * 70)
        print("BP + Decimation Graph Coloring")
        print("=" * 70)
        print(f"Map: {map_name}")
        print(f"Nodes: {nodes}, Edges: {edges}")
        print(f"Colors: {num_colors}, Penalty: {config.same_color_penalty}")
        print(f"BP: max_iter={config.max_iter}, tol={config.tolerance}, "
              f"damping={config.damping}, schedule={config.update_schedule}")
        print(f"\n" + "=" * 70)
        print("Decimation Process")
        print("=" * 70)
    
    # Decimation 循环
    evidence = {}
    unfixed_nodes = list(nodes)
    step = 0
    history = []
    
    while unfixed_nodes:
        step += 1
        
        # 运行 BP 获取所有节点的 belief
        all_beliefs = engine.run_all_beliefs(mn, evidence, trace=True, trace_beliefs=True)
        bp_trace = engine.last_trace
        bp_trace_records = bp_trace.to_list() if bp_trace else []
        bp_metrics = _summarize_bp_trace(bp_trace)
        map_conflicts = compute_map_conflicts(
            edges, all_beliefs, evidence=evidence, domains=mn.var_domains
        )
        
        # 收集候选节点
        candidates = []
        for node in nodes:
            if node in evidence:
                continue
            probs = all_beliefs[node]
            best_color = int(np.argmax(probs))
            max_prob = probs[best_color]
            candidates.append((node, best_color, max_prob))
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_node, best_color, best_prob = candidates[0]
        
        # 对称性检测与破缺
        if best_prob <= threshold:
            target_node = random.choice(unfixed_nodes)
            target_color = random.choice(color_domain)
            action = f"Random: {target_node}={target_color}"
            is_symmetric = True
        else:
            target_node, target_color = best_node, best_color
            action = f"Confident: {target_node}={target_color}"
            is_symmetric = False
        
        # 记录历史
        history.append({
            'step': step,
            'beliefs': all_beliefs.copy(),
            'evidence': evidence.copy(),
            'action': action,
            'target': (target_node, target_color),
            'unfixed': unfixed_nodes.copy(),
            'is_symmetric': is_symmetric,
            'bp_iter': engine.last_iter,
            'bp_residual': engine.last_residual,
            'bp_converged': engine.last_converged,
            'bp_trace': bp_trace_records,
            'MAP_conflicts': map_conflicts,
            'bp_metrics': bp_metrics,
        })
        
        # 更新 evidence
        evidence[target_node] = target_color
        unfixed_nodes.remove(target_node)
        
        if verbose:
            evidence_str = ", ".join([f"{k}={v}" for k, v in evidence.items()])
            print(f"[Step {step}] {action}")
            print(f"          Evidence: {{{evidence_str}}}")
            print(f"          BP: iter={engine.last_iter}, "
                  f"residual={engine.last_residual:.2e}, "
                  f"converged={engine.last_converged}")
            if show_bp_trace:
                mean_h_last = bp_metrics["mean_entropy_curve"][-1] if bp_metrics["mean_entropy_curve"] else float("nan")
                print(f"          BP-trace: flips={bp_metrics['total_flips']}, "
                      f"nodes_with_flips={bp_metrics['nodes_with_flips']}, "
                      f"max_flips/node={bp_metrics['max_flips_per_node']}, "
                      f"mean_H(last)={mean_h_last:.3f}")
            print(f"          MAP_conflicts={map_conflicts}")
    
    # 记录最终状态
    final_beliefs = engine.run_all_beliefs(mn, evidence, trace=True, trace_beliefs=True)
    final_bp_trace = engine.last_trace
    final_bp_trace_records = final_bp_trace.to_list() if final_bp_trace else []
    final_bp_metrics = _summarize_bp_trace(final_bp_trace)
    final_map_conflicts = compute_map_conflicts(
        edges, final_beliefs, evidence=evidence, domains=mn.var_domains
    )
    
    history.append({
        'step': step + 1,
        'beliefs': final_beliefs,
        'evidence': evidence.copy(),
        'action': 'Complete',
        'target': None,
        'unfixed': [],
        'is_symmetric': False,
        'bp_iter': engine.last_iter,
        'bp_residual': engine.last_residual,
        'bp_converged': engine.last_converged,
        'bp_trace': final_bp_trace_records,
        'MAP_conflicts': final_map_conflicts,
        'bp_metrics': final_bp_metrics,
    })
    
    # 验证结果
    conflicts = [(u, v) for u, v in edges if evidence[u] == evidence[v]]
    
    return {
        'map_name': map_name,
        'num_colors': num_colors,
        'nodes': nodes,
        'edges': edges,
        'evidence': evidence,
        'history': history,
        'conflicts': conflicts,
        'solved': len(conflicts) == 0,
        'steps': step,
    }


def print_belief_table(result: dict):
    """打印 Belief 变化表格"""
    history = result['history']
    nodes = result['nodes']
    
    print(f"\n[Belief Evolution Table]")
    print("-" * 70)
    
    header = f"{'Step':<6}"
    for node in sorted(nodes):
        header += f"{node:^20}"
    print(header)
    print("-" * 70)
    
    for record in history:
        row = f"{record['step']:<6}"
        for node in sorted(nodes):
            probs = record['beliefs'][node]
            prob_str = "[" + ",".join([f"{p:.2f}" for p in probs]) + "]"
            if node in record['evidence']:
                prob_str += f"={record['evidence'][node]}"
            row += f"{prob_str:^20}"
        print(row)


def print_result(result: dict):
    """打印求解结果"""
    print(f"\n" + "=" * 70)
    print("Result")
    print("=" * 70)
    
    evidence = result['evidence']
    nodes = result['nodes']
    conflicts = result['conflicts']
    edges = result['edges']
    
    sorted_nodes = sorted(nodes)
    print(f"Coloring: " + " -> ".join([f"{n}({evidence[n]})" for n in sorted_nodes]))
    
    if conflicts:
        print(f"X Conflicts: {conflicts} ({len(conflicts)}/{len(edges)})")
    else:
        print(f"V No conflicts! Success in {result['steps']} steps")
    
    final_metrics = result['history'][-1]
    print(f"MAP_conflicts (belief+evidence): {final_metrics['MAP_conflicts']}")


def visualize_decimation(result: dict, save_path: Optional[str] = None):
    """可视化 Decimation 过程"""
    history = result['history']
    nodes = result['nodes']
    edges = result['edges']
    num_colors = result['num_colors']
    map_name = result['map_name']
    
    # 构建 networkx 图
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42, k=1.5)
    
    threshold = 1.0 / num_colors + 0.1
    
    n_steps = len(history)
    cols = min(n_steps, 4)
    rows = (n_steps + cols - 1) // cols
    
    fig = plt.figure(figsize=(6 * cols, 6 * rows))
    
    for idx, record in enumerate(history):
        ax = fig.add_subplot(rows, cols, idx + 1)
        
        # 绘制边
        for u, v in G.edges():
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            
            if (u in record['evidence'] and v in record['evidence'] and 
                record['evidence'][u] == record['evidence'][v]):
                ax.plot(x, y, 'r-', linewidth=4, alpha=0.6, zorder=1)
            else:
                ax.plot(x, y, 'gray', linewidth=2, alpha=0.4, zorder=1)
        
        # 绘制节点
        for node in G.nodes():
            x, y = pos[node]
            probs = record['beliefs'][node]
            max_prob = np.max(probs)
            size = 0.08 + 0.04 * max_prob
            
            if node in record['evidence']:
                c = record['evidence'][node]
                color = COLOR_MAP[c % len(COLOR_MAP)]
                circle = plt.Circle((x, y), size, color=color, ec='black', linewidth=2, zorder=3)
                ax.add_patch(circle)
                ax.text(x, y - size - 0.15, f"{node}={record['evidence'][node]}", 
                       ha='center', va='top', fontsize=12, fontweight='bold')
            else:
                if max_prob <= threshold:
                    for i, prob in enumerate(probs):
                        if prob > 0.01:
                            theta1 = 360 * sum(probs[:i]) / sum(probs)
                            theta2 = 360 * sum(probs[:i+1]) / sum(probs)
                            wedge = mpatches.Wedge((x, y), size, theta1, theta2, 
                                                  facecolor=COLOR_MAP[i % len(COLOR_MAP)],
                                                  edgecolor='red', linewidth=3, zorder=3)
                            ax.add_patch(wedge)
                    prob_text = ",".join([f"{p:.2f}" for p in probs])
                    ax.text(x, y - size - 0.15, f"{node}\n[{prob_text}]", 
                           ha='center', va='top', fontsize=10, color='red', fontweight='bold')
                else:
                    best_color = np.argmax(probs)
                    color = COLOR_MAP[best_color % len(COLOR_MAP)]
                    circle = plt.Circle((x, y), size, color=color, ec='green', 
                                      linewidth=3, alpha=0.8, zorder=3)
                    ax.add_patch(circle)
                    ax.text(x, y - size - 0.15, f"{node}\n->{best_color}({max_prob:.2f})", 
                           ha='center', va='top', fontsize=10, color='green', fontweight='bold')
        
        # 标题
        if record['target']:
            target_node, target_color = record['target']
            if record['is_symmetric']:
                title = f"Step {record['step']}: Random Break - {target_node}={target_color}"
                title_color = 'red'
            else:
                title = f"Step {record['step']}: Fix Confident - {target_node}={target_color}"
                title_color = 'green'
        else:
            title = f"Step {record['step']}: Complete"
            title_color = 'black'
        
        ax.set_title(title, fontsize=13, fontweight='bold', color=title_color, pad=15)
        ax.set_xlim(min(p[0] for p in pos.values()) - 0.3, max(p[0] for p in pos.values()) + 0.3)
        ax.set_ylim(min(p[1] for p in pos.values()) - 0.4, max(p[1] for p in pos.values()) + 0.3)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # 图例
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_MAP[0], edgecolor='black', label='Color 0'),
        mpatches.Patch(facecolor=COLOR_MAP[1], edgecolor='black', label='Color 1'),
        mpatches.Patch(facecolor='white', edgecolor='red', linewidth=3, label='Symmetric State'),
        mpatches.Patch(facecolor='white', edgecolor='green', linewidth=3, label='Confident State'),
        mpatches.Patch(facecolor='white', edgecolor='black', linewidth=2, label='Fixed Node'),
        plt.Line2D([0], [0], color='red', linewidth=4, label='Conflict Edge')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.01), 
               ncol=6, fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    plt.suptitle(f"Decimation Process - {map_name} ({num_colors} Colors)", 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nVisualization saved: {save_path}")
    
    plt.show()


def visualize_bp_evolution(result: dict, config: BPConfig, save_path: Optional[str] = None):
    """可视化 BP 内部迭代演化"""
    history = result['history']
    nodes = result['nodes']
    num_colors = result['num_colors']
    map_name = result['map_name']
    
    meaningful_steps = [h for h in history if h['target'] is not None]
    
    if not meaningful_steps:
        return
    
    n_steps = len(meaningful_steps)
    fig, axes = plt.subplots(n_steps, 2, figsize=(14, 4 * n_steps))
    if n_steps == 1:
        axes = axes.reshape(1, -1)
    
    for idx, record in enumerate(meaningful_steps):
        bp_trace = record.get('bp_trace', [])
        bp_metrics = record.get('bp_metrics', {})
        unfixed = record.get('unfixed', [])
        step_num = record['step']
        
        # 左图：Residual 曲线
        ax_residual = axes[idx, 0]
        residual_curve = bp_metrics.get('residual_curve', [])
        if residual_curve:
            iterations = list(range(1, len(residual_curve) + 1))
            ax_residual.semilogy(iterations, residual_curve, 'b-o', markersize=4, linewidth=1.5)
            ax_residual.axhline(y=config.tolerance, color='r', linestyle='--', alpha=0.7, 
                               label=f'Tolerance={config.tolerance}')
            ax_residual.set_xlabel('Iteration', fontsize=11)
            ax_residual.set_ylabel('Residual (log scale)', fontsize=11)
            ax_residual.set_title(f'Step {step_num}: BP Convergence', fontsize=12, fontweight='bold')
            ax_residual.legend(loc='upper right', fontsize=9)
            ax_residual.grid(True, alpha=0.3)
        else:
            ax_residual.text(0.5, 0.5, 'No BP trace data', ha='center', va='center', fontsize=12)
            ax_residual.set_title(f'Step {step_num}: BP Convergence', fontsize=12, fontweight='bold')
        
        # 右图：代表节点的 belief 演化
        ax_belief = axes[idx, 1]
        
        if 'O' in unfixed:
            representative = 'O'
        elif unfixed:
            representative = unfixed[0]
        else:
            representative = None
        
        if representative and bp_trace:
            belief_history = []
            for rec in bp_trace:
                beliefs = rec.get('beliefs', {})
                if representative in beliefs:
                    probs = _safe_normalize(np.array(beliefs[representative]))
                    belief_history.append(probs)
            
            if belief_history:
                belief_array = np.array(belief_history)
                iterations = list(range(1, len(belief_history) + 1))
                
                for c in range(min(num_colors, belief_array.shape[1])):
                    ax_belief.plot(iterations, belief_array[:, c], 
                                  color=COLOR_MAP[c % len(COLOR_MAP)],
                                  marker='o', markersize=4, linewidth=1.5,
                                  label=f'Color {c}')
                
                ax_belief.axhline(y=1.0/num_colors, color='gray', linestyle=':', alpha=0.5, label='Uniform')
                ax_belief.set_xlabel('Iteration', fontsize=11)
                ax_belief.set_ylabel('P(color)', fontsize=11)
                ax_belief.set_title(f'Step {step_num}: Belief Evolution of Node "{representative}"', 
                                   fontsize=12, fontweight='bold')
                ax_belief.legend(loc='upper right', fontsize=9, ncol=2)
                ax_belief.set_ylim(-0.05, 1.05)
                ax_belief.grid(True, alpha=0.3)
                
                final_belief = belief_history[-1]
                best_c = int(np.argmax(final_belief))
                ax_belief.annotate(f'Final: Color {best_c} ({final_belief[best_c]:.2f})',
                                  xy=(len(iterations), final_belief[best_c]),
                                  xytext=(len(iterations) - 1, final_belief[best_c] + 0.1),
                                  fontsize=10, color=COLOR_MAP[best_c % len(COLOR_MAP)],
                                  fontweight='bold',
                                  arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
            else:
                ax_belief.text(0.5, 0.5, f'No belief data for {representative}', 
                              ha='center', va='center', fontsize=12)
                ax_belief.set_title(f'Step {step_num}: Belief Evolution', fontsize=12, fontweight='bold')
        else:
            ax_belief.text(0.5, 0.5, 'No unfixed nodes', ha='center', va='center', fontsize=12)
            ax_belief.set_title(f'Step {step_num}: Belief Evolution', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'BP Internal Iteration Details - {map_name} ({num_colors} Colors)',
                fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"BP internal evolution saved: {save_path}")
    
    plt.show()


# ============================================================================
# 演示函数
# ============================================================================

def demo_basic():
    """基本演示：pentagon 地图"""
    print("=" * 70)
    print("演示 1：基本 BP + Decimation（pentagon 地图）")
    print("=" * 70)
    
    result = run_decimation("pentagon", num_colors=4)
    print_belief_table(result)
    print_result(result)
    
    return result


def demo_triangle():
    """三角形地图演示"""
    print("\n" + "=" * 70)
    print("演示 2：三角形地图（3色）")
    print("=" * 70)
    
    result = run_decimation("triangle", num_colors=3)
    print_result(result)
    
    return result


def demo_australia():
    """澳大利亚地图演示"""
    print("\n" + "=" * 70)
    print("演示 3：澳大利亚地图")
    print("=" * 70)
    
    result = run_decimation("australia")
    print_result(result)
    
    return result


def demo_compare_maps():
    """比较不同地图"""
    print("\n" + "=" * 70)
    print("演示 4：多地图比较")
    print("=" * 70)
    
    maps_to_test = ["triangle", "square", "pentagon", "australia"]
    
    print(f"\n{'Map':<15} {'Colors':<8} {'Steps':<8} {'Conflicts':<10} {'Result'}")
    print("-" * 60)
    
    for map_name in maps_to_test:
        map_config = get_map(map_name)
        num_colors = len(map_config["colors"])
        
        result = run_decimation(map_name, verbose=False)
        
        status = "✓ Success" if result['solved'] else f"✗ {len(result['conflicts'])} conflicts"
        print(f"{map_name:<15} {num_colors:<8} {result['steps']:<8} {len(result['conflicts']):<10} {status}")


if __name__ == "__main__":
    # 运行基本演示
    result = demo_basic()
    
    # 可视化
    if SHOW_PLOT:
        visualize_decimation(result)
        
        config = BPConfig()
        visualize_bp_evolution(result, config)
    
    # 其他演示
    demo_triangle()
    demo_australia()
    demo_compare_maps()
