"""
Message Flow Visualization for Belief Propagation.

This module provides rich visualizations for BP message propagation,
showing how messages flow between nodes and how beliefs evolve.

Key features:
- Belief state visualization with color-coded confidence
- Message flow arrows showing information propagation direction
- Convergence tracking and entropy visualization
- Support for both tree and loopy BP
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import networkx as nx

from map_coloring.analysis.plots.base import DEFAULT_COLORS, get_color, safe_normalize, compute_entropy


def _compute_belief_confidence(belief: np.ndarray) -> Tuple[int, float, str]:
    """
    Compute confidence metrics for a belief distribution.
    
    Returns:
        (dominant_color, confidence_score, status_label)
        - confidence_score: 0 = uniform, 1 = certain
        - status_label: "Certain", "Confident", "Uncertain", "Uniform"
    """
    belief = safe_normalize(belief)
    n = len(belief)
    uniform_prob = 1.0 / n
    
    sorted_probs = np.sort(belief)[::-1]
    dominant = int(np.argmax(belief))
    max_prob = sorted_probs[0]
    second_prob = sorted_probs[1] if n > 1 else 0.0
    gap = max_prob - second_prob
    
    # Confidence score: how far from uniform
    # 1 - normalized entropy
    entropy = compute_entropy(belief)
    max_entropy = np.log(n)
    confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
    
    # Status label
    if max_prob > 0.95:
        status = "Certain"
    elif gap > 0.3:
        status = "Confident"
    elif max_prob > uniform_prob + 0.1:
        status = "Leaning"
    else:
        status = "Uncertain"
    
    return dominant, confidence, status


class MessageFlowPlotter:
    """Plotter for visualizing BP message flow with rich annotations."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
    
    def plot_belief_state(
        self,
        G: nx.Graph,
        beliefs: Dict[str, np.ndarray],
        evidence: Optional[Dict[str, int]] = None,
        iteration: int = 0,
        pos: Optional[Dict] = None,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        show_probs: bool = True,
        show_confidence: bool = True,
    ) -> plt.Axes:
        """
        Plot belief state with rich color-coded visualization.
        
        Args:
            G: NetworkX graph
            beliefs: Dict mapping node to belief array
            evidence: Dict of fixed nodes (shown with special border)
            iteration: Current iteration number
            pos: Node positions
            title: Plot title
            ax: Matplotlib axes
            show_probs: Show probability values
            show_confidence: Show confidence indicators
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        if pos is None:
            pos = nx.spring_layout(G, seed=42)
        
        evidence = evidence or {}
        
        # Calculate bounds from positions
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_range = max(x_max - x_min, 0.1)
        y_range = max(y_max - y_min, 0.1)
        margin = 0.25
        
        # Draw edges first (background)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#AAAAAA', width=2.5, alpha=0.6)
        
        # Calculate node size based on graph
        node_size = min(800, 3000 / max(len(G.nodes()), 1))
        font_size = max(8, min(12, 120 / max(len(G.nodes()), 1)))
        
        # Prepare node colors and styles
        node_colors = []
        node_edge_colors = []
        node_edge_widths = []
        node_alphas = []
        
        for node in G.nodes():
            if node in evidence:
                color_idx = evidence[node]
                node_colors.append(get_color(color_idx))
                node_edge_colors.append('black')
                node_edge_widths.append(4)
                node_alphas.append(1.0)
            elif node in beliefs:
                belief = np.array(beliefs[node])
                dominant, confidence, status = _compute_belief_confidence(belief)
                node_colors.append(get_color(dominant))
                if status == "Certain":
                    node_edge_colors.append('darkgreen')
                    node_edge_widths.append(3)
                elif status == "Confident":
                    node_edge_colors.append('green')
                    node_edge_widths.append(2.5)
                elif status == "Leaning":
                    node_edge_colors.append('orange')
                    node_edge_widths.append(2)
                else:
                    node_edge_colors.append('red')
                    node_edge_widths.append(2)
                node_alphas.append(0.5 + 0.5 * confidence)
            else:
                node_colors.append('lightgray')
                node_edge_colors.append('gray')
                node_edge_widths.append(1)
                node_alphas.append(0.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_size,
            edgecolors=node_edge_colors,
            linewidths=node_edge_widths,
            alpha=0.9
        )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size, font_weight='bold')
        
        # Add annotations
        offset_y = y_range * 0.12 # Increased offset to prevent overlap
        for node in G.nodes():
            x, y = pos[node]
            
            # Probability annotation (below node)
            if show_probs and node in beliefs and node not in evidence:
                belief = beliefs[node]
                
                # Check for uniform distribution
                n_colors = len(belief)
                if np.allclose(belief, 1.0/n_colors, atol=0.01):
                    display_text = "Uniform"
                else:
                    prob_str = ','.join([f'{p:.2f}' for p in belief])
                    display_text = f"[{prob_str}]"

                ax.text(x, y - offset_y, display_text, ha='center', va='top', 
                       fontsize=max(6, font_size - 2), color='#333333',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='#DDDDDD'),
                       zorder=10) # Ensure text is on top
            
            # Status annotation (above node)
            if show_confidence and node not in evidence and node in beliefs:
                belief = np.array(beliefs[node])
                _, _, status = _compute_belief_confidence(belief)
                if status == "Certain":
                    status_color = 'darkgreen'
                elif status == "Confident":
                    status_color = 'green'
                elif status == "Leaning":
                    status_color = 'orange'
                else:
                    status_color = 'red'
                ax.text(x, y + offset_y, status, ha='center', va='bottom',
                       fontsize=max(6, font_size - 2), color=status_color, style='italic')
            
            # Fixed indicator
            if node in evidence:
                ax.text(x, y + offset_y, f'Fixed={evidence[node]}', ha='center', va='bottom',
                       fontsize=max(6, font_size - 2), color='black', fontweight='bold')
        
        # Title
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'Belief State - Iteration {iteration}', fontsize=12, fontweight='bold')
        
        # Set axis limits with margin
        ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        ax.set_aspect('equal')
        ax.axis('off')
        
        return ax
    
    def plot_message_snapshot(
        self,
        G: nx.Graph,
        messages: Dict[Tuple[str, str], np.ndarray],
        beliefs: Dict[str, np.ndarray],
        iteration: int,
        pos: Optional[Dict] = None,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot a snapshot of messages at a given iteration.
        
        Args:
            G: NetworkX graph
            messages: Dict mapping (source, target) to message array
            beliefs: Dict mapping node to belief array
            iteration: Current iteration number
            pos: Node positions (computed if None)
            title: Plot title
            ax: Matplotlib axes
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        if pos is None:
            pos = nx.spring_layout(G, seed=42)
        
        # Draw edges with message strength
        edge_colors = []
        edge_widths = []
        
        for u, v in G.edges():
            # Get messages in both directions
            msg_uv = messages.get((u, v), None)
            msg_vu = messages.get((v, u), None)
            
            # Calculate message "strength" as entropy
            strength = 0
            count = 0
            for msg in [msg_uv, msg_vu]:
                if msg is not None:
                    msg_arr = np.array(msg)
                    # Lower entropy = stronger message
                    msg_norm = msg_arr / (msg_arr.sum() + 1e-10)
                    entropy = -np.sum(msg_norm * np.log(msg_norm + 1e-10))
                    max_ent = np.log(len(msg_arr)) if len(msg_arr) > 1 else 1
                    strength += 1 - entropy / max_ent
                    count += 1
            
            if count > 0:
                strength /= count
            
            edge_colors.append(strength)
            edge_widths.append(1 + strength * 3)
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_colors,
            edge_cmap=plt.cm.Reds,
            width=edge_widths,
            alpha=0.7
        )
        
        # Draw nodes with belief colors
        node_colors = []
        for node in G.nodes():
            belief = beliefs.get(node, None)
            if belief is not None:
                dominant = np.argmax(belief)
                node_colors.append(get_color(dominant))
            else:
                node_colors.append('lightgray')
        
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=500,
            edgecolors='black',
            linewidths=2
        )
        
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
        
        # Add belief annotations
        for node in G.nodes():
            belief = beliefs.get(node, None)
            if belief is not None:
                x, y = pos[node]
                belief_str = ','.join([f'{p:.2f}' for p in belief])
                ax.annotate(
                    f'[{belief_str}]',
                    (x, y - 0.15),
                    ha='center',
                    fontsize=8,
                    color='blue'
                )
        
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'Message Flow - Iteration {iteration}', fontsize=12, fontweight='bold')
        
        ax.axis('off')
        return ax


def plot_message_flow_snapshot(
    G: nx.Graph,
    messages: Dict[Tuple[str, str], np.ndarray],
    beliefs: Dict[str, np.ndarray],
    iteration: int,
    pos: Optional[Dict] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Axes:
    """
    Convenience function to plot a message flow snapshot.
    
    Args:
        G: NetworkX graph
        messages: Dict mapping (source, target) to message array  
        beliefs: Dict mapping node to belief array
        iteration: Current iteration number
        pos: Node positions
        title: Plot title
        ax: Matplotlib axes
        figsize: Figure size
        
    Returns:
        Matplotlib axes
    """
    plotter = MessageFlowPlotter(figsize=figsize)
    return plotter.plot_message_snapshot(
        G, messages, beliefs, iteration, pos, title, ax
    )


def plot_belief_state(
    G: nx.Graph,
    beliefs: Dict[str, np.ndarray],
    evidence: Optional[Dict[str, int]] = None,
    iteration: int = 0,
    pos: Optional[Dict] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 8),
    show_probs: bool = True,
    show_confidence: bool = True,
) -> plt.Axes:
    """
    Convenience function to plot belief state with rich visualization.
    """
    plotter = MessageFlowPlotter(figsize=figsize)
    return plotter.plot_belief_state(
        G, beliefs, evidence, iteration, pos, title, ax, show_probs, show_confidence
    )


def create_message_flow_animation(
    G: nx.Graph,
    bp_trace: List[Dict],
    output_path: str,
    pos: Optional[Dict] = None,
    figsize: Tuple[int, int] = (10, 8),
    interval: int = 1000,
    evidence: Optional[Dict[str, int]] = None,
) -> str:
    """
    Create an animated GIF showing message flow over iterations.
    
    Args:
        G: NetworkX graph
        bp_trace: List of iteration records with 'messages' and 'beliefs' keys
        output_path: Path to save the GIF
        pos: Node positions
        figsize: Figure size
        interval: Time between frames in ms
        evidence: Fixed nodes to highlight
        
    Returns:
        Path to saved animation
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("Animation requires matplotlib with Pillow. Saving static frames instead.")
        return _save_static_frames(G, bp_trace, output_path, pos, figsize)
    
    if not bp_trace:
        print("Warning: No BP trace data available for animation")
        return output_path
    
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    fig, ax = plt.subplots(figsize=figsize)
    plotter = MessageFlowPlotter(figsize=figsize)
    
    def update(frame):
        ax.clear()
        record = bp_trace[frame]
        beliefs = record.get('beliefs', {})
        iteration = record.get('iter', record.get('iteration', frame + 1))
        residual = record.get('residual', 0.0)
        converged = record.get('converged', False)
        
        # Use belief state visualization for richer display
        plotter.plot_belief_state(
            G, beliefs, evidence, iteration, pos,
            title=f'BP Iteration {iteration} | Residual: {residual:.4f} | {"Converged" if converged else "Running"}',
            ax=ax
        )
        return ax,
    
    anim = FuncAnimation(
        fig, update, frames=len(bp_trace),
        interval=interval, blit=False
    )
    
    try:
        anim.save(output_path, writer=PillowWriter(fps=max(1, 1000 // interval)))
        print(f"Animation saved to: {output_path}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
    
    plt.close(fig)
    
    return output_path


def _save_static_frames(
    G: nx.Graph,
    bp_trace: List[Dict],
    output_path: str,
    pos: Optional[Dict] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> str:
    """Save static frames as PNG files."""
    import os
    
    base_path = output_path.rsplit('.', 1)[0]
    os.makedirs(base_path, exist_ok=True)
    
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    plotter = MessageFlowPlotter(figsize=figsize)
    
    for i, record in enumerate(bp_trace):
        fig, ax = plt.subplots(figsize=figsize)
        beliefs = record.get('beliefs', {})
        iteration = record.get('iter', record.get('iteration', i + 1))
        
        plotter.plot_belief_state(
            G, beliefs, None, iteration, pos,
            title=f'BP Iteration {iteration}',
            ax=ax
        )
        
        frame_path = os.path.join(base_path, f'frame_{i:03d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    return base_path


def visualize_message_propagation_steps(
    G: nx.Graph,
    bp_trace: List[Dict],
    max_steps: int = 6,
    pos: Optional[Dict] = None,
    figsize: Tuple[int, int] = (16, 10),
    evidence: Optional[Dict[str, int]] = None,
) -> plt.Figure:
    """
    Create a grid visualization showing message propagation steps.
    
    Args:
        G: NetworkX graph
        bp_trace: List of iteration records
        max_steps: Maximum number of steps to show
        pos: Node positions
        figsize: Figure size
        evidence: Fixed nodes to highlight
        
    Returns:
        Matplotlib figure
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    if not bp_trace:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No BP trace data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    n_steps = min(len(bp_trace), max_steps)
    n_cols = min(3, n_steps)
    n_rows = (n_steps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    plotter = MessageFlowPlotter()
    
    # Select evenly spaced iterations
    if len(bp_trace) <= max_steps:
        indices = list(range(len(bp_trace)))
    else:
        indices = np.linspace(0, len(bp_trace) - 1, max_steps, dtype=int).tolist()
    
    for idx, trace_idx in enumerate(indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        record = bp_trace[trace_idx]
        beliefs = record.get('beliefs', {})
        iteration = record.get('iter', record.get('iteration', trace_idx + 1))
        residual = record.get('residual', 0.0)
        converged = record.get('converged', False)
        
        status = "✓" if converged else f"r={residual:.3f}"
        plotter.plot_belief_state(
            G, beliefs, evidence, iteration, pos,
            title=f'Iter {iteration} ({status})',
            ax=ax,
            show_confidence=True,
            show_probs=True,
        )
    
    # Hide empty subplots
    for idx in range(len(indices), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    fig.suptitle('BP Message Propagation Steps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_message_comparison(
    G: nx.Graph,
    pure_bp_trace: List[Dict],
    decimation_trace: List[Dict],
    pos: Optional[Dict] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Compare message flow between Pure BP and BP with Decimation.
    
    Args:
        G: NetworkX graph
        pure_bp_trace: Trace from pure BP
        decimation_trace: Trace from BP with decimation
        pos: Node positions
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    plotter = MessageFlowPlotter()
    
    # Plot pure BP final state
    if pure_bp_trace:
        final_pure = pure_bp_trace[-1]
        plotter.plot_belief_state(
            G, 
            final_pure.get('beliefs', {}),
            None,
            final_pure.get('iter', final_pure.get('iteration', len(pure_bp_trace))),
            pos,
            title='Pure BP (Final)',
            ax=ax1
        )
    else:
        ax1.text(0.5, 0.5, 'No Pure BP data', ha='center', va='center')
        ax1.axis('off')
    
    # Plot decimation final state
    if decimation_trace:
        final_dec = decimation_trace[-1]
        plotter.plot_belief_state(
            G,
            final_dec.get('beliefs', {}),
            None,
            final_dec.get('iter', final_dec.get('iteration', len(decimation_trace))),
            pos,
            title='BP + Decimation (Final)',
            ax=ax2
        )
    else:
        ax2.text(0.5, 0.5, 'No Decimation data', ha='center', va='center')
        ax2.axis('off')
    
    fig.suptitle('Pure BP vs BP with Decimation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# =============================================================================
# Rich Pure BP Visualization Functions
# =============================================================================

def _build_graph_from_result(result: Dict[str, Any]) -> nx.Graph:
    """Build NetworkX graph from run_pure_bp result."""
    G = nx.Graph()
    G.add_nodes_from(result.get('nodes', []))
    G.add_edges_from(result.get('edges', []))
    return G


def visualize_pure_bp_result(
    result: Dict[str, Any],
    save_path: Optional[str] = None,
    show: bool = True,
    max_iters: int = 6,
    layout: Optional[Dict] = None,
) -> plt.Figure:
    """
    Visualize Pure BP result with rich multi-panel display.
    
    Shows:
    - Belief propagation steps (grid of iterations)
    - Final belief state with confidence indicators
    - Convergence summary
    
    Args:
        result: Result dict from run_pure_bp()
        save_path: Path to save the figure
        show: Whether to display the figure
        max_iters: Maximum number of iterations to show
        layout: Node positions dict
        
    Returns:
        Matplotlib figure
    """
    G = _build_graph_from_result(result)
    bp_trace = result.get('bp_trace', [])
    evidence = result.get('evidence', {})
    final_beliefs = result.get('beliefs', {})
    
    pos = layout
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    # Create comprehensive visualization
    fig = _create_pure_bp_comprehensive_figure(
        G, bp_trace, evidence, final_beliefs, result, pos, max_iters
    )
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Pure BP visualization to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def _create_pure_bp_comprehensive_figure(
    G: nx.Graph,
    bp_trace: List[Dict],
    evidence: Dict[str, int],
    final_beliefs: Dict[str, List],
    result: Dict[str, Any],
    pos: Dict,
    max_iters: int,
) -> plt.Figure:
    """Create a comprehensive Pure BP visualization figure."""
    
    num_colors = result.get('num_colors', 3)
    map_name = result.get('map_name', 'Unknown')
    n_trace = len(bp_trace)
    
    plotter = MessageFlowPlotter()
    
    if n_trace == 0:
        # No trace - just show final state with summary
        fig = plt.figure(figsize=(14, 6))
        
        ax1 = fig.add_subplot(1, 2, 1)
        plotter.plot_belief_state(
            G, final_beliefs, evidence, 0, pos,
            title='Final Belief State',
            ax=ax1
        )
        
        ax2 = fig.add_subplot(1, 2, 2)
        _plot_bp_summary(ax2, result, final_beliefs)
        
        fig.suptitle(f'Pure BP Result - {map_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    if n_trace == 1:
        # Tree structure - single iteration, show before/after comparison
        # Use larger figure size to prevent overlapping
        fig = plt.figure(figsize=(22, 9))
        
        # Left: Initial state (uniform)
        ax1 = fig.add_subplot(1, 3, 1)
        uniform_beliefs = {n: [1.0/num_colors] * num_colors for n in G.nodes() if n not in evidence}
        # Add evidence nodes with delta distribution
        for node, color in evidence.items():
            delta = [0.0] * num_colors
            delta[color] = 1.0
            uniform_beliefs[node] = delta
        plotter.plot_belief_state(
            G, uniform_beliefs, evidence, 0, pos,
            title='Step 0: Initial State\n(Before BP)',
            ax=ax1,
            show_confidence=False,
        )
        
        # Middle: After BP
        ax2 = fig.add_subplot(1, 3, 2)
        plotter.plot_belief_state(
            G, final_beliefs, evidence, 1, pos,
            title='Step 1: After BP\n(Converged in 1 iteration)',
            ax=ax2
        )
        
        # Right: Summary + Legend
        ax3 = fig.add_subplot(1, 3, 3)
        _plot_bp_summary_with_legend(ax3, result, final_beliefs, num_colors)
        
        fig.suptitle(f'Pure BP on Tree Structure - {map_name}\n(Exact inference converges in 1 pass)', 
                    fontsize=16, fontweight='bold')
                    
        # Add explanation note at the bottom
        fig.text(0.5, 0.02, 
                 f"Note: [p0, p1, ...] indicates probabilities for Color 0, Color 1, etc. 'Uniform' means equal probability.", 
                 ha='center', fontsize=12, style='italic', color='#444444')
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        return fig
    
    # Multiple iterations - show grid of iterations
    n_show = min(n_trace, max_iters)
    n_cols = min(3, n_show)
    n_rows = (n_show + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(6 * n_cols, 6 * n_rows + 2))
    
    # Select iterations to show
    if n_trace <= max_iters:
        indices = list(range(n_trace))
    else:
        indices = np.linspace(0, n_trace - 1, max_iters, dtype=int).tolist()
    
    # Plot iteration steps
    for idx, trace_idx in enumerate(indices):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        record = bp_trace[trace_idx]
        beliefs = record.get('beliefs', {})
        iteration = record.get('iter', record.get('iteration', trace_idx + 1))
        residual = record.get('residual', 0.0)
        converged = record.get('converged', False)
        
        if converged:
            status = "✓ Converged"
        else:
            status = f"Residual: {residual:.4f}"
        
        plotter.plot_belief_state(
            G, beliefs, evidence, iteration, pos,
            title=f'Iteration {iteration}\n{status}',
            ax=ax
        )
    
    # Hide unused subplot cells
    for idx in range(len(indices), n_rows * n_cols):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        ax.axis('off')
    
    fig.suptitle(f'Pure BP Message Propagation - {map_name}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def _plot_bp_summary_with_legend(ax: plt.Axes, result: Dict[str, Any], beliefs: Dict[str, List], num_colors: int):
    """Plot BP summary with color legend."""
    ax.axis('off')
    
    # Gather statistics
    map_name = result.get('map_name', 'Unknown')
    num_nodes = len(result.get('nodes', []))
    num_edges = len(result.get('edges', []))
    bp_iter = result.get('bp_iter', 0)
    bp_converged = result.get('bp_converged', False)
    bp_residual = result.get('bp_residual', 0.0)
    elapsed = result.get('elapsed_seconds', 0.0)
    evidence = result.get('evidence', {})
    
    # Compute belief statistics
    entropies = []
    confidences = []
    for node, belief in beliefs.items():
        if node not in evidence:
            belief_arr = np.array(belief)
            entropies.append(compute_entropy(belief_arr))
            _, conf, _ = _compute_belief_confidence(belief_arr)
            confidences.append(conf)
    
    avg_entropy = np.mean(entropies) if entropies else 0
    avg_confidence = np.mean(confidences) if confidences else 0
    max_entropy = np.log(num_colors) if num_colors > 0 else 1
    
    # Summary text
    summary_lines = [
        f"Graph: {num_nodes} nodes, {num_edges} edges",
        f"Colors: {num_colors}",
        f"Evidence: {evidence}",
        "",
        f"BP Iterations: {bp_iter}",
        f"Converged: {'Yes' if bp_converged else 'No'}",
        f"Final Residual: {bp_residual:.6f}",
        "",
        f"Avg Entropy: {avg_entropy:.3f} / {max_entropy:.3f}",
        f"Avg Confidence: {avg_confidence:.1%}",
        f"Time: {elapsed:.3f}s",
    ]
    
    # Draw summary box
    summary_text = '\n'.join(summary_lines)
    ax.text(0.5, 0.65, summary_text, transform=ax.transAxes,
           ha='center', va='center', fontsize=10,
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9, edgecolor='gray'))
    
    # Draw color legend
    ax.text(0.5, 0.2, 'Color Legend:', transform=ax.transAxes,
           ha='center', va='top', fontsize=10, fontweight='bold')
    
    legend_y = 0.12
    for c in range(num_colors):
        color = get_color(c)
        rect = plt.Rectangle((0.3 + c * 0.15, legend_y), 0.1, 0.06, 
                             facecolor=color, edgecolor='black', linewidth=1,
                             transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.35 + c * 0.15, legend_y - 0.02, str(c), transform=ax.transAxes,
               ha='center', va='top', fontsize=9)


def _plot_bp_summary(ax: plt.Axes, result: Dict[str, Any], beliefs: Dict[str, List]):
    """Plot BP summary statistics."""
    ax.axis('off')
    
    # Gather statistics
    map_name = result.get('map_name', 'Unknown')
    num_nodes = len(result.get('nodes', []))
    num_edges = len(result.get('edges', []))
    num_colors = result.get('num_colors', 0)
    bp_iter = result.get('bp_iter', 0)
    bp_converged = result.get('bp_converged', False)
    bp_residual = result.get('bp_residual', 0.0)
    elapsed = result.get('elapsed_seconds', 0.0)
    evidence = result.get('evidence', {})
    
    # Compute belief statistics
    entropies = []
    confidences = []
    for node, belief in beliefs.items():
        if node not in evidence:
            belief_arr = np.array(belief)
            entropies.append(compute_entropy(belief_arr))
            _, conf, _ = _compute_belief_confidence(belief_arr)
            confidences.append(conf)
    
    avg_entropy = np.mean(entropies) if entropies else 0
    avg_confidence = np.mean(confidences) if confidences else 0
    max_entropy = np.log(num_colors) if num_colors > 0 else 1
    
    # Create summary text
    summary_lines = [
        f"Map: {map_name} | Nodes: {num_nodes} | Edges: {num_edges} | Colors: {num_colors}",
        f"Evidence: {evidence if evidence else 'None'}",
        f"BP Iterations: {bp_iter} | Converged: {'Yes' if bp_converged else 'No'} | Final Residual: {bp_residual:.6f}",
        f"Avg Entropy: {avg_entropy:.3f} / {max_entropy:.3f} | Avg Confidence: {avg_confidence:.2%}",
        f"Time: {elapsed:.3f}s",
    ]
    
    summary_text = '\n'.join(summary_lines)
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
           ha='center', va='center', fontsize=10,
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))


def visualize_belief_evolution_pure_bp(
    result: Dict[str, Any],
    target_nodes: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize how beliefs evolve over BP iterations for specific nodes.
    
    Args:
        result: Result dict from run_pure_bp()
        target_nodes: Nodes to track (default: all non-evidence nodes)
        save_path: Path to save the figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    bp_trace = result.get('bp_trace', [])
    evidence = result.get('evidence', {})
    num_colors = result.get('num_colors', 3)
    nodes = result.get('nodes', [])
    
    if not bp_trace:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No BP trace data available', ha='center', va='center')
        ax.axis('off')
        return fig
    
    # Select nodes to track
    if target_nodes is None:
        target_nodes = [n for n in nodes if n not in evidence][:6]  # Max 6 nodes
    
    n_nodes = len(target_nodes)
    if n_nodes == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No non-evidence nodes to track', ha='center', va='center')
        ax.axis('off')
        return fig
    
    n_cols = min(3, n_nodes)
    n_rows = (n_nodes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes = axes.flatten()
    
    for idx, node in enumerate(target_nodes):
        ax = axes[idx]
        
        # Extract belief history for this node
        iterations = []
        belief_history = []
        
        for record in bp_trace:
            beliefs = record.get('beliefs', {})
            if node in beliefs:
                iterations.append(record.get('iter', record.get('iteration', len(iterations) + 1)))
                belief_history.append(np.array(beliefs[node]))
        
        if not belief_history:
            ax.text(0.5, 0.5, f'No data for {node}', ha='center', va='center')
            ax.set_title(f'Node: {node}')
            continue
        
        belief_array = np.array(belief_history)
        
        # Plot each color's probability over iterations
        for c in range(min(num_colors, belief_array.shape[1])):
            ax.plot(iterations, belief_array[:, c], 
                   color=get_color(c), marker='o', markersize=4,
                   linewidth=2, label=f'Color {c}')
        
        ax.axhline(y=1.0/num_colors, color='gray', linestyle=':', alpha=0.5, label='Uniform')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('P(color)')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Node: {node}', fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_nodes, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Belief Evolution - {result.get("map_name", "Unknown")}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved belief evolution to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def create_pure_bp_animation(
    result: Dict[str, Any],
    save_path: str,
    duration_sec: float = 5,
    layout: Optional[Dict] = None,
) -> str:
    """
    Create animation from Pure BP result.
    
    Args:
        result: Result dict from run_pure_bp()
        save_path: Path to save the GIF
        duration_sec: Total duration in seconds
        layout: Node positions dict
        
    Returns:
        Path to saved animation
    """
    G = _build_graph_from_result(result)
    bp_trace = result.get('bp_trace', [])
    evidence = result.get('evidence', {})
    
    if not bp_trace:
        print("Warning: No BP trace data available for animation")
        # Create a single-frame "animation" with final state
        fig, ax = plt.subplots(figsize=(10, 8))
        plotter = MessageFlowPlotter()
        plotter.plot_belief_state(
            G, result.get('beliefs', {}), evidence, 0, layout,
            title='Pure BP Final State',
            ax=ax
        )
        fig.savefig(save_path.replace('.gif', '.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved static image to: {save_path.replace('.gif', '.png')}")
        return save_path.replace('.gif', '.png')
    
    pos = layout
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    interval = int(duration_sec * 1000 / max(len(bp_trace), 1))
    
    return create_message_flow_animation(
        G, bp_trace, save_path, pos=pos, interval=interval, evidence=evidence
    )
