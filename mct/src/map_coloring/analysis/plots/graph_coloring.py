# -*- coding: utf-8 -*-
"""
图着色状态可视化（通用模块）
支持 BP、CSP 等不同算法的着色结果展示
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from map_coloring.analysis.plots.base import DEFAULT_COLORS, get_color, safe_normalize


def build_graph(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    layout: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]]]:
    """
    构建 networkx 图并计算布局
    
    Args:
        nodes: 节点列表
        edges: 边列表
        layout: 可选的预定义布局 {node: (x, y)}，如果不提供则自动计算
    """
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    if layout is not None:
        pos = layout
    else:
        pos = nx.spring_layout(G, seed=42, k=1.5)
    
    return G, pos


def plot_coloring_state(
    ax: plt.Axes,
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    assignment: Dict[str, int],
    beliefs: Optional[Dict[str, np.ndarray]] = None,
    num_colors: int = 4,
    title: str = "",
    title_color: str = "black",
    highlight_node: Optional[str] = None,
    highlight_style: str = "confident",  # "confident", "random", "fixed", "initial"
    show_beliefs: bool = True,
    node_size: float = 0.1,
    palette: Optional[List[str]] = None,
    threshold_margin: float = 0.1,
    confidence_gap: float = 0.0,
    initial_nodes: Optional[List[str]] = None,
):
    """
    绘制单个着色状态
    
    Args:
        ax: matplotlib axes
        G: networkx 图
        pos: 节点位置
        assignment: 当前赋值 {node: color_index}
        beliefs: 节点的 belief 分布 {node: prob_array}
        num_colors: 颜色数量
        title: 标题
        title_color: 标题颜色
        highlight_node: 高亮的节点
        highlight_style: 高亮样式
        show_beliefs: 是否显示 belief 信息
        node_size: 节点大小
        palette: 颜色调色板
    """
    colors = palette or DEFAULT_COLORS
    threshold = 1.0 / num_colors + float(threshold_margin)
    initial_node_set = set(initial_nodes or [])
    
    # 绘制边
    for u, v in G.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        
        # 检查是否冲突（两个已赋值节点同色）
        if (u in assignment and v in assignment and assignment[u] == assignment[v]):
            ax.plot(x, y, 'r-', linewidth=4, alpha=0.6, zorder=1)
        else:
            ax.plot(x, y, 'gray', linewidth=2, alpha=0.4, zorder=1)
    
    # 绘制节点
    for node in G.nodes():
        x, y = pos[node]
        probs = beliefs.get(node) if beliefs else None
        max_prob = float(np.max(probs)) if probs is not None else 0.0
        size = node_size + 0.03 * max_prob
        
        is_highlight = (node == highlight_node)
        
        if node in assignment:
            # 已赋值节点
            c = assignment[node]
            fill_color = get_color(c, colors)
            
            # 特殊处理：如果是当前步骤的高亮节点 (刚刚被赋值)，
            # 我们希望同时看到之前的 Belief (原因) 和现在的赋值 (结果)
            # 只有当 beliefs 存在且显示 beliefs 时才这样处理
            if is_highlight and beliefs and show_beliefs and probs is not None:
                 # 画饼图 (Pre-condition)
                if len(probs) >= 2:
                    sorted_probs = np.sort(probs)
                    gap = float(sorted_probs[-1] - sorted_probs[-2])
                else:
                    gap = float(max_prob)

                # 判断是用饼图还是实心圆 (根据之前的状态)
                threshold = 1.0 / num_colors + float(threshold_margin)
                if max_prob <= threshold or gap < float(confidence_gap):
                     # 之前是对称状态：画饼图
                    for i, prob in enumerate(probs):
                        if prob > 0.01:
                            theta1 = 360 * sum(probs[:i]) / sum(probs)
                            theta2 = 360 * sum(probs[:i+1]) / sum(probs)
                            wedge = mpatches.Wedge(
                                (x, y), size, theta1, theta2,
                                facecolor=get_color(i, colors),
                                edgecolor='none', zorder=3
                            )
                            ax.add_patch(wedge)
                    prob_text = ",".join([f"{p:.2f}" for p in probs])
                else:
                    # 之前就已经很确信了：画实心圆 (但也可能是错色，不过这里我们画最可能的分布色)
                    # 为了可视化一致性，如果它之前很确信，那基本上就是那个颜色
                    # 这里依然画分布的主要颜色，或者干脆画成实心
                    circle = plt.Circle((x, y), size, color=fill_color, zorder=3)
                    ax.add_patch(circle)

                # 用粗边框表示“选中了” (Post-condition)
                if highlight_style == "random":
                    ec = 'red'
                elif highlight_style == "confident":
                    ec = 'green'
                else:
                    ec = 'black'
                
                # 画一个空心的圈来表示边框
                circle_outline = plt.Circle((x, y), size, fill=False, ec=ec, linewidth=4, zorder=4)
                ax.add_patch(circle_outline)

                # 文字显示：赋值结果
                ax.text(x, y - size - 0.12, f"{node}={assignment[node]}", 
                    ha='center', va='top', fontsize=10, fontweight='bold', color=ec)
                
                # 文字显示：之前的概率分布 (在上方或下方额外显示，为了不重叠，我们放在更下方)
                if max_prob <= threshold or gap < float(confidence_gap):
                     ax.text(x, y + size + 0.05, f"[{prob_text}]", 
                       ha='center', va='bottom', fontsize=8, color='red')

            else:
                # 普通已赋值节点 (历史节点)
                # 初始固定节点用紫色边框
                if node in initial_node_set:
                    ec = 'purple'
                    lw = 4
                elif is_highlight: # Should not happen if logic above is correct, but fallback
                    if highlight_style == "random":
                        ec = 'red'
                        lw = 4
                    elif highlight_style == "confident":
                        ec = 'green'
                        lw = 4
                    else:
                        ec = 'black'
                        lw = 2
                else:
                    ec = 'black'
                    lw = 2
                
                circle = plt.Circle((x, y), size, color=fill_color, ec=ec, linewidth=lw, zorder=3)
                ax.add_patch(circle)
                ax.text(x, y - size - 0.12, f"{node}={assignment[node]}", 
                    ha='center', va='top', fontsize=10, fontweight='bold')
        
        elif probs is not None and show_beliefs:
            # 未赋值但有 belief
            # 对称/不置信：max 不高，或 top1-top2 间隔太小
            if len(probs) >= 2:
                sorted_probs = np.sort(probs)
                gap = float(sorted_probs[-1] - sorted_probs[-2])
            else:
                gap = float(max_prob)

            if max_prob <= threshold or gap < float(confidence_gap):
                # 对称状态：画饼图
                for i, prob in enumerate(probs):
                    if prob > 0.01:
                        theta1 = 360 * sum(probs[:i]) / sum(probs)
                        theta2 = 360 * sum(probs[:i+1]) / sum(probs)
                        wedge = mpatches.Wedge(
                            (x, y), size, theta1, theta2,
                            facecolor=get_color(i, colors),
                            edgecolor='red', linewidth=2, zorder=3
                        )
                        ax.add_patch(wedge)
                prob_text = ",".join([f"{p:.2f}" for p in probs])
                ax.text(x, y - size - 0.12, f"{node}\n[{prob_text}]", 
                       ha='center', va='top', fontsize=9, color='red')
            else:
                # 有倾向：显示最可能的颜色
                best_color = int(np.argmax(probs))
                fill_color = get_color(best_color, colors)
                circle = plt.Circle((x, y), size, color=fill_color, ec='green', 
                                   linewidth=2, alpha=0.7, zorder=3)
                ax.add_patch(circle)
                ax.text(x, y - size - 0.12, f"{node}\n->{best_color}({max_prob:.2f})", 
                       ha='center', va='top', fontsize=9, color='green')
        else:
            # 未赋值无 belief
            circle = plt.Circle((x, y), size, color='lightgray', ec='gray', 
                               linewidth=2, linestyle='--', zorder=3)
            ax.add_patch(circle)
            ax.text(x, y - size - 0.12, f"{node}=?", 
                   ha='center', va='top', fontsize=9, color='gray')
    
    ax.set_title(title, fontsize=11, fontweight='bold', color=title_color, pad=10)
    ax.set_xlim(min(p[0] for p in pos.values()) - 0.3, max(p[0] for p in pos.values()) + 0.3)
    ax.set_ylim(min(p[1] for p in pos.values()) - 0.4, max(p[1] for p in pos.values()) + 0.3)
    ax.set_aspect('equal')
    ax.axis('off')


def create_coloring_legend(num_colors: int, palette: Optional[List[str]] = None) -> List[mpatches.Patch]:
    """创建着色图例"""
    colors = palette or DEFAULT_COLORS
    elements = []
    for i in range(min(num_colors, len(colors))):
        elements.append(mpatches.Patch(facecolor=colors[i], edgecolor='black', label=f'Color {i}'))
    
    elements.extend([
        mpatches.Patch(facecolor='white', edgecolor='red', linewidth=2, label='Symmetric'),
        mpatches.Patch(facecolor='white', edgecolor='green', linewidth=2, label='Confident'),
        mpatches.Patch(facecolor='white', edgecolor='black', linewidth=2, label='Fixed'),
        mpatches.Patch(facecolor='white', edgecolor='purple', linewidth=3, label='Initial'),
        plt.Line2D([0], [0], color='red', linewidth=3, label='Conflict Edge'),
    ])
    return elements
