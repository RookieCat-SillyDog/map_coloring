# -*- coding: utf-8 -*-
"""
Pure BP（纯置信传播）入口 - 不进行 Decimation，仅展示消息传播过程。

用于对比展示 BP 的并行消息传播特性与搜索算法的串行赋值差异。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Any, Tuple, List

import numpy as np
from pgm_toolkit.models.markovnet import MarkovNetwork

from map_coloring.agents.bayesian.research_bp import (
    InstrumentedBeliefPropagation,
    InstrumentedMaxProductBeliefPropagation,
    BPTrace,
)
from map_coloring.agents.bayesian.decimation import BPDecimationConfig, build_markov_network


@dataclass
class PureBPConfig:
    """Pure BP 配置"""
    max_iter: int = 100
    tolerance: float = 1e-3
    damping: float = 0.0
    update_schedule: str = "synchronous"
    rng_seed: int = 0
    same_color_penalty: float = 1e-6
    diff_color_reward: float = 1.0
    # 使用 Sum-Product (marginal) 还是 Max-Product (MAP)
    use_max_product: bool = False


def run_pure_bp(
    map_name: str,
    nodes: Iterable[str],
    edges: Iterable[Tuple[str, str]],
    num_colors: int,
    config: Optional[PureBPConfig] = None,
    initial_evidence: Optional[Dict[str, int]] = None,
    verbose: bool = True,
    trace_beliefs: bool = True,
    trace_messages: bool = True,
) -> Dict[str, Any]:
    """
    运行纯 BP（不进行 Decimation），用于展示消息传播过程。
    
    与 BP+Decimation 的区别：
    - 不会固定任何新节点（除了 initial_evidence）
    - 完整记录 BP 迭代过程中的消息传播
    - 适合展示"信息如何从 evidence 向全图扩散"
    
    Args:
        map_name: 地图名称
        nodes: 节点列表
        edges: 边列表
        num_colors: 颜色数
        config: BP 配置
        initial_evidence: 初始固定的节点赋值（用于打破对称性并观察传播）
        verbose: 是否打印过程
        trace_beliefs: 是否记录每轮迭代的信念
        trace_messages: 是否记录每轮迭代的消息（用于可视化消息流）
        
    Returns:
        结构化结果字典，包含：
        - beliefs: 最终信念分布
        - bp_trace: BP 迭代轨迹（包含消息快照）
        - evidence: 使用的证据
        - 元信息
    """
    cfg = config or PureBPConfig()
    
    nodes_list = list(nodes)
    edges_list = list(edges)
    
    # 构建 Markov 网络（复用 decimation 的构建函数）
    decimation_cfg = BPDecimationConfig(
        max_iter=cfg.max_iter,
        tolerance=cfg.tolerance,
        damping=cfg.damping,
        update_schedule=cfg.update_schedule,
        rng_seed=cfg.rng_seed,
        same_color_penalty=cfg.same_color_penalty,
        diff_color_reward=cfg.diff_color_reward,
    )
    mn = build_markov_network(nodes_list, edges_list, num_colors, decimation_cfg)
    
    # 选择 BP 引擎
    if cfg.use_max_product:
        engine = InstrumentedMaxProductBeliefPropagation(
            max_iter=cfg.max_iter,
            tol=cfg.tolerance,
            damping=cfg.damping,
            update_schedule=cfg.update_schedule,
            rng_seed=cfg.rng_seed,
        )
    else:
        engine = InstrumentedBeliefPropagation(
            max_iter=cfg.max_iter,
            tol=cfg.tolerance,
            damping=cfg.damping,
            update_schedule=cfg.update_schedule,
            rng_seed=cfg.rng_seed,
        )
    mn.set_inference_engine(engine)
    
    # 设置 evidence
    evidence: Dict[str, int] = {}
    if initial_evidence:
        evidence.update(initial_evidence)
    
    start_ts = time.time()
    
    if verbose:
        print(f"=== Pure BP (No Decimation) ===")
        print(f"Map={map_name}, nodes={len(nodes_list)}, edges={len(edges_list)}, colors={num_colors}")
        print(f"BP: max_iter={cfg.max_iter}, tol={cfg.tolerance}, damping={cfg.damping}")
        print(f"Update schedule: {cfg.update_schedule}")
        print(f"Engine: {'Max-Product' if cfg.use_max_product else 'Sum-Product'}")
        if evidence:
            print(f"Initial evidence: {evidence}")
        else:
            print("No initial evidence (fully symmetric)")
    
    # 运行 BP
    beliefs = engine.run_all_beliefs(
        mn, 
        evidence, 
        trace=True, 
        trace_beliefs=trace_beliefs,
        trace_messages=trace_messages,
    )
    
    elapsed = time.time() - start_ts
    
    # 获取 trace
    bp_trace: Optional[BPTrace] = engine.last_trace
    
    if verbose:
        print(f"\nBP completed in {elapsed:.3f}s")
        print(f"Iterations: {engine.last_iter}, Converged: {engine.last_converged}")
        print(f"Final residual: {engine.last_residual:.6f}")
        
        # 打印最终 beliefs 摘要
        print("\nFinal beliefs:")
        for node in sorted(nodes_list):
            probs = beliefs[node]
            probs_str = ", ".join([f"{p:.3f}" for p in probs])
            max_color = int(np.argmax(probs))
            max_prob = float(np.max(probs))
            status = f"[{max_color}]" if max_prob > 0.9 else f"{max_color}:{max_prob:.2f}"
            print(f"  {node}: [{probs_str}] -> {status}")
    
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
            "use_max_product": cfg.use_max_product,
        },
        "evidence": evidence,
        "beliefs": {k: list(v) for k, v in beliefs.items()},
        "bp_trace": bp_trace.to_list() if bp_trace else [],
        "bp_iter": engine.last_iter,
        "bp_residual": engine.last_residual,
        "bp_converged": engine.last_converged,
        "elapsed_seconds": elapsed,
    }


__all__ = ["PureBPConfig", "run_pure_bp"]
