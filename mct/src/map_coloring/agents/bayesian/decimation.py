# -*- coding: utf-8 -*-
"""
Belief Propagation + Decimation solver (agent layer).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Any, Tuple

import numpy as np
from pgm_toolkit.models.markovnet import MarkovNetwork

from map_coloring.agents.bayesian.research_bp import (
    InstrumentedMaxProductBeliefPropagation,
    compute_map_conflicts,
)


@dataclass
class BPDecimationConfig:
    max_iter: int = 100
    tolerance: float = 1e-3
    damping: float = 0.0
    update_schedule: str = "synchronous"
    rng_seed: int = 0
    same_color_penalty: float = 1e-6
    diff_color_reward: float = 1.0
    threshold_margin: float = 0.1
    confidence_gap: float = 0.2

    def threshold(self, num_colors: int) -> float:
        return 1.0 / num_colors + self.threshold_margin


def _safe_normalize(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(probs, 1.0 / len(probs), dtype=float)
    return probs / total


def build_markov_network(
    nodes: Iterable[str],
    edges: Iterable[Tuple[str, str]],
    num_colors: int,
    config: BPDecimationConfig,
) -> MarkovNetwork:
    mn = MarkovNetwork(name="map_coloring")
    color_domain = list(range(num_colors))

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
    nodes: Iterable[str],
    edges: Iterable[Tuple[str, str]],
    num_colors: int,
    config: Optional[BPDecimationConfig] = None,
    verbose: bool = True,
    trace_bp: bool = False,
    initial_evidence: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    运行 BP + Decimation，返回结构化结果。
    
    关键设计：无论同步还是异步更新，beliefs 总是全局一致的快照，
    这样绘图模块不需要知道更新方式。

    Args:
        map_name: 地图名称
        nodes: 节点列表
        edges: 边列表
        num_colors: 颜色数
        config: BP/decimation 超参
        verbose: 是否打印过程
        trace_bp: 是否记录 BP 内部迭代轨迹
        initial_evidence: 初始固定的节点赋值
    """
    cfg = config or BPDecimationConfig()
    threshold = cfg.threshold(num_colors)
    
    nodes_list = list(nodes)
    edges_list = list(edges)

    mn = build_markov_network(nodes_list, edges_list, num_colors, cfg)
    engine = InstrumentedMaxProductBeliefPropagation(
        max_iter=cfg.max_iter,
        tol=cfg.tolerance,
        damping=cfg.damping,
        update_schedule=cfg.update_schedule,
        rng_seed=cfg.rng_seed,
    )
    mn.set_inference_engine(engine)

    # 初始化 evidence
    evidence: Dict[str, int] = {}
    initial_nodes: list[str] = []
    if initial_evidence:
        evidence.update(initial_evidence)
        initial_nodes = list(initial_evidence.keys())
    
    unfixed_nodes = [n for n in nodes_list if n not in evidence]
    history = []
    step = 0

    start_ts = time.time()

    if verbose:
        print(f"Map={map_name}, nodes={len(nodes_list)}, edges={len(edges_list)}, colors={num_colors}")
        print(f"BP: max_iter={cfg.max_iter}, tol={cfg.tolerance}, damping={cfg.damping}")
        print(f"Update schedule: {cfg.update_schedule}")
        print(f"Decimation: threshold_margin={cfg.threshold_margin}, confidence_gap={cfg.confidence_gap}")
        if initial_evidence:
            print(f"Initial evidence: {initial_evidence}")

    # Run initial BP to capture the start state (Step 0)
    # This ensures we see the effect of initial_evidence before any decimation decision
    current_beliefs = engine.run_all_beliefs(mn, evidence, trace=trace_bp, trace_beliefs=trace_bp)
    
    initial_record = {
        "step": 0,
        "target": None,
        "confident": True,
        "best_prob": 0.0,
        "best_gap": 0.0,
        "beliefs": {k: list(v) for k, v in current_beliefs.items()},
        "evidence": evidence.copy(),
        "unfixed": unfixed_nodes.copy(),
        "map_conflicts": compute_map_conflicts(edges_list, current_beliefs, evidence=evidence, domains=mn.var_domains),
        "bp_iter": engine.last_iter,
        "bp_residual": engine.last_residual,
        "bp_converged": engine.last_converged,
    }
    history.append(initial_record)

    while unfixed_nodes:
        step += 1

        # 运行 BP 推断（可能是同步或异步）
        # 如果需要 trace，这一步会记录中间过程
        beliefs_with_trace = engine.run_all_beliefs(mn, evidence, trace=trace_bp, trace_beliefs=trace_bp)
        
        # Capture trace before it gets overwritten by subsequent calls
        current_trace = engine.last_trace

        # 关键：无论同步还是异步，都再做一次同步采样确保全局一致性
        # 这样绘图模块就不需要知道更新方式
        beliefs_snapshot = engine.run_all_beliefs(mn, evidence, trace=False, trace_beliefs=False)

        # 选择最有把握的节点（基于同步快照）
        candidates = []
        for node in nodes_list:
            if node in evidence:
                continue
            probs = _safe_normalize(np.asarray(beliefs_snapshot[node]))
            sorted_probs = np.sort(probs)[::-1]
            best_color = int(np.argmax(probs))
            best_prob = float(sorted_probs[0])
            second_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
            gap = best_prob - second_prob
            candidates.append((node, best_color, best_prob, gap))

        candidates.sort(key=lambda x: x[2], reverse=True)
        best_node, best_color, best_prob, best_gap = candidates[0]

        # 判断是否真正 Confident
        if best_prob <= threshold or best_gap < cfg.confidence_gap:
            # Random 模式：随机选择节点，但颜色要基于 BP 的概率分布采样
            # 这样可以避免选择被约束排除的颜色（概率接近 0 的颜色）
            target_node = random.choice(unfixed_nodes)
            
            # 获取该节点的概率分布，基于概率采样颜色
            node_probs = _safe_normalize(np.asarray(beliefs_snapshot[target_node]))
            # 使用概率分布采样，而不是均匀随机
            target_color = int(np.random.choice(num_colors, p=node_probs))
            confident = False
        else:
            target_node, target_color = best_node, best_color
            confident = True

        evidence[target_node] = target_color
        unfixed_nodes.remove(target_node)

        map_conflicts = compute_map_conflicts(edges_list, beliefs_snapshot, evidence=evidence, domains=mn.var_domains)

        step_record: Dict[str, Any] = {
            "step": step,
            "target": (target_node, target_color),
            "confident": confident,
            "best_prob": best_prob,
            "best_gap": best_gap,
            "beliefs": {k: list(v) for k, v in beliefs_snapshot.items()},  # 使用同步快照
            "evidence": evidence.copy(),
            "unfixed": unfixed_nodes.copy(),
            "map_conflicts": map_conflicts,
            "bp_iter": engine.last_iter,
            "bp_residual": engine.last_residual,
            "bp_converged": engine.last_converged,
        }
        
        if trace_bp:
            # Use the captured trace from the first run (engine.last_trace might be overwritten)
            step_record["bp_trace"] = current_trace.to_list() if current_trace else []
        
        history.append(step_record)

        if verbose:
            state = "Confident" if confident else "Random"
            print(f"[Step {step:02d}] {state}: {target_node}={target_color} | "
                  f"prob={best_prob:.3f}, gap={best_gap:.3f} | conflicts={map_conflicts}")

    elapsed = time.time() - start_ts
    final_conflicts = [(u, v) for u, v in edges_list if evidence[u] == evidence[v]]

    return {
        "map_name": map_name,
        "nodes": nodes_list,
        "edges": edges_list,
        "num_colors": num_colors,
        "config": {
            "max_iter": cfg.max_iter,
            "tolerance": cfg.tolerance,
            "damping": cfg.damping,
            "update_schedule": cfg.update_schedule,
            "threshold_margin": cfg.threshold_margin,
            "confidence_gap": cfg.confidence_gap,
        },
        "initial_nodes": initial_nodes,
        "evidence": evidence,
        "history": history,
        "conflicts": final_conflicts,
        "solved": len(final_conflicts) == 0,
        "steps": step,
        "elapsed_seconds": elapsed,
    }


__all__ = ["BPDecimationConfig", "run_decimation", "build_markov_network"]
