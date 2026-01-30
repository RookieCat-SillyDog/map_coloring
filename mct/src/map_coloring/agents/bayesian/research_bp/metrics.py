"""
BP（置信传播）消减实验的度量辅助工具。

该模块提供了用于评估图模型上置信传播算法性能的度量函数，
包括计算期望冲突数和MAP（最大后验概率）冲突数。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from pgm_toolkit.core.graph import Factor


def _as_prob_array(belief: Any) -> np.ndarray:
    """
    将信念转换为概率数组。
    
    参数:
        belief: 输入的信念，可以是Factor对象或其他可转换为数组的对象
        
    返回:
        转换后的概率数组，数据类型为float
    """
    if isinstance(belief, Factor):
        return np.asarray(belief.table, dtype=float)
    return np.asarray(belief, dtype=float)


def compute_expected_violations(
    edges: Iterable[Tuple[Any, Any]],
    beliefs: Dict[Any, Any],
) -> float:
    """
    计算期望冲突数。
    
    期望冲突数定义为：sum_{(i,j)} sum_c b_i(c) b_j(c)
    其中(i,j)是边，c是可能的取值，b_i(c)是节点i取值为c的信念概率
    
    参数:
        edges: 图中的边的集合，每条边表示为(节点1, 节点2)的元组
        beliefs: 字典，键为节点，值为该节点的信念分布
        
    返回:
        期望冲突数的浮点值
    """
    total = 0.0
    # 遍历所有边
    for u, v in edges:
        # 如果边的任一端点没有信念信息，跳过
        if u not in beliefs or v not in beliefs:
            continue
        # 将信念转换为概率数组
        bu = _as_prob_array(beliefs[u])
        bv = _as_prob_array(beliefs[v])
        # 计算信念概率的总和
        bu_sum = float(np.sum(bu))
        bv_sum = float(np.sum(bv))
        # 归一化信念概率
        if bu_sum > 0:
            bu = bu / bu_sum
        if bv_sum > 0:
            bv = bv / bv_sum
        # 计算两个信念向量的点积，即两个节点取相同值的概率之和
        total += float(np.dot(bu, bv))
    return float(total)


def compute_map_conflicts(
    edges: Iterable[Tuple[Any, Any]],
    beliefs: Dict[Any, Any],
    evidence: Optional[Dict[Any, Any]] = None,
    domains: Optional[Dict[Any, List[Any]]] = None,
) -> int:
    """
    计算从证据和MAP完成中产生的冲突数。
    
    MAP（Maximum A Posteriori）完成是指为每个变量选择具有最大后验概率的值。
    冲突是指相邻节点被分配了相同的值。
    
    参数:
        edges: 图中的边的集合，每条边表示为(节点1, 节点2)的元组
        beliefs: 字典，键为节点，值为该节点的信念分布
        evidence: 可选，已知证据的字典，键为节点，值为该节点的已知取值
        domains: 可选，字典，键为节点，值为该节点可能的取值列表
        
    返回:
        冲突数的整数值
    """
    evidence = evidence or {}
    # 初始化赋值字典，包含所有已知证据
    assignment: Dict[Any, Any] = dict(evidence)

    # 为每个没有证据的节点选择MAP估计
    for var, belief in beliefs.items():
        # 如果节点已有证据，跳过
        if var in assignment:
            continue
        # 如果信念是Factor对象
        if isinstance(belief, Factor):
            # 从Factor对象中获取变量域
            domain = belief.var_domains[var]
            # 选择信念表中最大概率对应的域值
            assignment[var] = domain[int(np.argmax(belief.table))]
        else:
            # 如果信念是数组形式
            if domains is not None and var in domains:
                # 使用提供的域信息
                domain = domains[var]
                assignment[var] = domain[int(np.argmax(belief))]
            else:
                # 直接使用最大概率的索引作为赋值
                assignment[var] = int(np.argmax(belief))

    # 计算冲突数
    conflicts = 0
    for u, v in edges:
        # 如果边的两个端点都有赋值且赋值相同，则冲突数加1
        if u in assignment and v in assignment and assignment[u] == assignment[v]:
            conflicts += 1
    return conflicts


class MetricsCalculator:
    """
    度量计算器类，用于计算图模型的各种度量指标。
    
    该类封装了图的边信息，并提供了计算多种度量指标的统一接口。
    """
    
    def __init__(self, edges: Iterable[Tuple[Any, Any]]) -> None:
        """
        初始化度量计算器。
        
        参数:
            edges: 图中的边的集合，每条边表示为(节点1, 节点2)的元组
        """
        self.edges = list(edges)

    def compute(
        self,
        beliefs: Dict[Any, Any],
        evidence: Optional[Dict[Any, Any]] = None,
        domains: Optional[Dict[Any, List[Any]]] = None,
    ) -> Dict[str, Any]:
        """
        计算所有度量指标。
        
        参数:
            beliefs: 字典，键为节点，值为该节点的信念分布
            evidence: 可选，已知证据的字典，键为节点，值为该节点的已知取值
            domains: 可选，字典，键为节点，值为该节点可能的取值列表
            
        返回:
            包含各种度量指标的字典，键为度量名称，值为度量值
            包括:
                - "E_viol": 期望冲突数
                - "MAP_conflicts": MAP冲突数
        """
        return {
            "E_viol": compute_expected_violations(self.edges, beliefs),
            "MAP_conflicts": compute_map_conflicts(
                self.edges, beliefs, evidence=evidence, domains=domains
            ),
        }