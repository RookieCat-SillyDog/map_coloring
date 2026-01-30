"""
带跟踪功能的置信传播（BP）引擎。

这是pgm_toolkit BP引擎的本地副本，具有可选的跟踪功能，
以及一次性计算所有变量信念的辅助函数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pgm_toolkit.core.graph import Factor
from pgm_toolkit.inference.exact_inference.core import InferenceEngine


@dataclass(frozen=True)
class _FactorNodeId:
    """Internal factor-node identifier used in the bipartite factor graph.

    Avoids collisions with user variable IDs (e.g. a variable literally named
    ``"f_0"``) and avoids assuming variable identifiers are strings.
    """

    idx: int

    def __repr__(self) -> str:
        return f"f_{self.idx}"


@dataclass
class BPIterationRecord:
    """
    BP迭代记录类，用于存储每次迭代的信息。
    
    属性:
        iteration: 迭代次数
        residual: 残差，表示消息变化的最大值
        converged: 是否收敛
        beliefs: 可选，当前迭代的信念分布
        messages: 可选，当前迭代的消息（用于可视化消息传播）
        updated_edges: 可选，本轮更新的边列表
    """
    iteration: int
    residual: float
    converged: bool
    beliefs: Optional[Dict[Any, np.ndarray]] = None
    messages: Optional[Dict[Tuple[Any, Any], np.ndarray]] = None
    updated_edges: Optional[List[Tuple[Any, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        将记录转换为字典格式。
        
        返回:
            包含记录信息的字典
        """
        data = {
            "iter": int(self.iteration),
            "residual": float(self.residual),
            "converged": bool(self.converged),
        }
        if self.beliefs is not None:
            data["beliefs"] = {
                var: np.asarray(vals, dtype=float).tolist()
                for var, vals in self.beliefs.items()
            }
        if self.messages is not None:
            # 转换消息为可序列化格式：(from, to) -> [probs]
            data["messages"] = {
                f"{src}->{dst}": np.asarray(vals, dtype=float).tolist()
                for (src, dst), vals in self.messages.items()
            }
        if self.updated_edges is not None:
            data["updated_edges"] = [(str(u), str(v)) for u, v in self.updated_edges]
        return data


@dataclass
class BPTrace:
    """
    BP跟踪类，用于存储BP算法的完整执行轨迹。
    
    属性:
        records: 迭代记录列表
        max_iter: 最大迭代次数
        tol: 收敛容忍度
    """
    records: List[BPIterationRecord] = field(default_factory=list)
    max_iter: int = 0
    tol: float = 0.0

    def to_list(self) -> List[Dict[str, Any]]:
        """
        将轨迹转换为字典列表。
        
        返回:
            包含所有迭代记录的字典列表
        """
        return [record.to_dict() for record in self.records]


class InstrumentedBeliefPropagation(InferenceEngine):
    """
    带跟踪功能的置信传播引擎类。
    
    该类实现了置信传播算法，并添加了跟踪功能，可以记录算法的执行过程。
    """
    
    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        damping: float = 0.0,
        update_schedule: str = "synchronous",
        rng_seed: Optional[int] = None,
    ) -> None:
        """
        初始化置信传播引擎。
        
        参数:
            max_iter: 循环BP的最大迭代次数
            tol: 消息收敛的容忍度（最大L∞差异）
        """
        self.max_iter = max_iter
        self.tol = tol

        if not (0.0 <= float(damping) < 1.0):
            raise ValueError("damping must be in [0.0, 1.0).")
        self.damping = float(damping)

        self.update_schedule = str(update_schedule).lower().strip()
        if self.update_schedule not in {"synchronous", "asynchronous", "random"}:
            raise ValueError(
                "update_schedule must be one of: 'synchronous', 'asynchronous', 'random'."
            )

        self.rng_seed = rng_seed
        self._rng = np.random.default_rng(rng_seed)

        # 存储最后一次运行的信息
        self.last_iter: int = 0
        self.last_residual: float = 0.0
        self.last_converged: bool = True
        self.last_trace: Optional[BPTrace] = None

    # ------------------------------------------------------------------
    # 因子图构建/结构辅助函数
    # ------------------------------------------------------------------
    def _build_factor_graph(
        self, model
    ) -> Tuple[Dict[Any, Dict], Dict[_FactorNodeId, Dict], Dict[Any, set]]:
        """
        从模型的因子构建二分因子图。
        
        参数:
            model: 概率图模型
            
        返回:
            包含变量节点、因子节点和边的元组
        """
        variable_nodes: Dict[Any, Dict] = {}
        factor_nodes: Dict[_FactorNodeId, Dict] = {}
        edges: Dict[Any, set] = {}

        # 从所有因子中的变量创建变量节点
        for factor in model.get_factors():
            for var in factor.scope:
                if var not in variable_nodes:
                    if hasattr(model, "var_domains") and var in model.var_domains:
                        domain = list(model.var_domains[var])
                    else:
                        domain = list(factor.var_domains[var])
                    variable_nodes[var] = {
                        "type": "variable",
                        "domain": domain,
                        "factors": set(),
                    }

        # 创建因子节点和无向边
        for idx, factor in enumerate(model.get_factors()):
            factor_id = _FactorNodeId(idx)
            factor_nodes[factor_id] = {
                "type": "factor",
                "factor": factor,
                "variables": set(factor.scope),
            }

            for var in factor.scope:
                edges.setdefault(var, set()).add(factor_id)
                edges.setdefault(factor_id, set()).add(var)
                variable_nodes[var]["factors"].add(factor_id)

        return variable_nodes, factor_nodes, edges

    def _is_tree(self, edges: Dict[Any, set]) -> bool:
        """
        检查（无向）因子图是否为单个树。
        
        参数:
            edges: 表示图的边字典
            
        返回:
            如果图是树则返回True，否则返回False
        """
        if not edges:
            return True

        visited = set()
        start = next(iter(edges.keys()))
        stack = [(start, None)]

        while stack:
            node, parent = stack.pop()
            if node in visited:
                return False  # 检测到环
            visited.add(node)
            for neigh in edges[node]:
                if neigh != parent:
                    stack.append((neigh, node))

        return len(visited) == len(edges)

    # ------------------------------------------------------------------
    # 消息计算辅助函数
    # ------------------------------------------------------------------
    def _compute_message_var_to_factor(
        self,
        var: Any,
        factor: Any,
        messages: Dict[Tuple[Any, Any], Factor],
        edges: Dict[Any, set],
        variable_nodes: Dict[Any, Dict],
        factor_nodes: Dict[Any, Dict],
    ) -> Factor:
        """
        计算从变量节点到因子节点的消息。
        
        参数:
            var: 变量节点名称
            factor: 因子节点名称
            messages: 当前消息字典
            edges: 图的边
            variable_nodes: 变量节点字典
            factor_nodes: 因子节点字典
            
        返回:
            计算出的消息（Factor对象）
        """
        # 获取与该变量相连的其他因子
        other_factors = [f for f in edges[var] if f != factor]

        # 如果没有其他因子，返回均匀消息
        if not other_factors:
            domain = variable_nodes[var]["domain"]
            table = np.ones(len(domain), dtype=float)
            return Factor([var], table, {var: domain}).normalize()

        # 收集来自其他因子的消息
        message_factors: List[Factor] = []
        for other_factor in other_factors:
            msg_key = (other_factor, var)
            if msg_key in messages:
                message_factors.append(messages[msg_key])

        # 如果没有消息，返回均匀消息
        if not message_factors:
            domain = variable_nodes[var]["domain"]
            table = np.ones(len(domain), dtype=float)
            return Factor([var], table, {var: domain}).normalize()

        # 将所有消息相乘并归一化
        result = Factor.multiply(message_factors)
        return result.normalize()

    def _compute_message_factor_to_var(
        self,
        factor: Any,
        var: Any,
        messages: Dict[Tuple[Any, Any], Factor],
        edges: Dict[Any, set],
        variable_nodes: Dict[Any, Dict],
        factor_nodes: Dict[Any, Dict],
    ) -> Factor:
        """
        计算从因子节点到变量节点的消息。
        
        参数:
            factor: 因子节点名称
            var: 变量节点名称
            messages: 当前消息字典
            edges: 图的边
            variable_nodes: 变量节点字典
            factor_nodes: 因子节点字典
            
        返回:
            计算出的消息（Factor对象）
        """
        factor_obj: Factor = factor_nodes[factor]["factor"]
        # 获取因子中除目标变量外的其他变量
        other_vars = [v for v in factor_obj.scope if v != var]

        # 如果没有其他变量，直接返回因子
        if not other_vars:
            return factor_obj.copy().normalize()

        # 收集因子和来自其他变量的消息
        message_factors: List[Factor] = [factor_obj]
        for other_var in other_vars:
            msg_key = (other_var, factor)
            if msg_key in messages:
                message_factors.append(messages[msg_key])

        # 将所有因子和消息相乘
        result = Factor.multiply(message_factors)

        # 对除目标变量外的所有变量求和（边缘化）
        for other_var in other_vars:
            if other_var in result.scope:
                result = result.sum_out(other_var)

        return result.normalize()

    # ------------------------------------------------------------------
    # 树结构BP
    # ------------------------------------------------------------------
    def _belief_propagation_tree(
        self,
        model,
        query_vars: List[Any],
        evidence: Dict[Any, Any],
    ) -> Dict[Any, Factor]:
        """
        在树结构上运行置信传播算法。
        
        参数:
            model: 概率图模型
            query_vars: 查询变量列表
            evidence: 证据字典
            
        返回:
            包含所有变量信念的字典
        """
        # 根据证据简化因子
        reduced_factors = [f.reduce(evidence) for f in model.get_factors()]

        if not reduced_factors:
            return {}

        # 创建临时模型类
        class TempModel:
            def __init__(self, factors):
                self._factors = factors

            def get_factors(self):
                return self._factors

        temp_model = TempModel(reduced_factors)
        variable_nodes, factor_nodes, edges = self._build_factor_graph(temp_model)

        if not variable_nodes:
            return {}

        messages: Dict[Tuple[Any, Any], Factor] = {}

        # 向上传递消息（从叶子到根）
        def dfs_up(node: Any, parent: Optional[Any] = None) -> None:
            for neigh in edges[node]:
                if neigh == parent:
                    continue
                dfs_up(neigh, node)

            if parent is not None:
                if node in factor_nodes:
                    msg = self._compute_message_factor_to_var(
                        node, parent, messages, edges, variable_nodes, factor_nodes
                    )
                else:
                    msg = self._compute_message_var_to_factor(
                        node, parent, messages, edges, variable_nodes, factor_nodes
                    )
                messages[(node, parent)] = msg

        # 修正：处理森林（多个连通分量）的情况
        # 1. 识别所有连通分量并为每个分量选择一个根
        roots = []
        visited_bfs = set()
        
        # 遍历所有节点以发现分量
        for node in variable_nodes:
            if node not in visited_bfs:
                roots.append(node)
                # BFS 标记该分量所有节点
                stack = [node]
                while stack:
                    curr = stack.pop()
                    if curr in visited_bfs:
                        continue
                    visited_bfs.add(curr)
                    for neigh in edges.get(curr, []):
                        if neigh not in visited_bfs:
                            stack.append(neigh)

        # 2. 对每个分量运行 Upward Pass
        for root in roots:
            dfs_up(root)

        # 向下传递消息（从根到叶子）
        def dfs_down(node: Any, parent: Optional[Any] = None) -> None:
            for neigh in edges[node]:
                if neigh == parent:
                    continue
                if node in factor_nodes:
                    msg = self._compute_message_factor_to_var(
                        node, neigh, messages, edges, variable_nodes, factor_nodes
                    )
                else:
                    msg = self._compute_message_var_to_factor(
                        node, neigh, messages, edges, variable_nodes, factor_nodes
                    )
                messages[(node, neigh)] = msg
                dfs_down(neigh, node)

        # 3. 对每个分量运行 Downward Pass
        for root in roots:
            dfs_down(root)

        return self._collect_beliefs(variable_nodes, edges, messages, evidence)

    # ------------------------------------------------------------------
    # 带跟踪的循环BP
    # ------------------------------------------------------------------
    def _belief_propagation_loopy(
        self,
        model,
        query_vars: List[Any],
        evidence: Dict[Any, Any],
        trace: bool = False,
        trace_beliefs: bool = False,
        trace_messages: bool = False,
        trace_callback: Optional[Callable[[BPIterationRecord], None]] = None,
    ) -> Dict[Any, Factor]:
        """
        在带环图上运行置信传播算法（循环BP）。
        
        参数:
            model: 概率图模型
            query_vars: 查询变量列表
            evidence: 证据字典
            trace: 是否记录跟踪信息
            trace_beliefs: 是否记录每轮迭代的信念
            trace_messages: 是否记录每轮迭代的消息（用于可视化消息传播）
            trace_callback: 可选的回调函数，每轮迭代后调用
            
        返回:
            包含所有变量信念的字典
        """
        # 根据证据简化因子
        reduced_factors = [f.reduce(evidence) for f in model.get_factors()]

        if not reduced_factors:
            return {}

        # 创建临时模型类
        class TempModel:
            def __init__(self, factors):
                self._factors = factors

            def get_factors(self):
                return self._factors

        temp_model = TempModel(reduced_factors)
        variable_nodes, factor_nodes, edges = self._build_factor_graph(temp_model)

        if not variable_nodes:
            return {}

        # 初始化消息为均匀分布
        messages: Dict[Tuple[Any, Any], Factor] = {}
        for node in edges:
            for neigh in edges[node]:
                if node in factor_nodes and neigh in variable_nodes:
                    domain = variable_nodes[neigh]["domain"]
                    table = np.ones(len(domain), dtype=float)
                    messages[(node, neigh)] = Factor(
                        [neigh], table, {neigh: domain}
                    ).normalize()
                elif node in variable_nodes and neigh in factor_nodes:
                    domain = variable_nodes[node]["domain"]
                    table = np.ones(len(domain), dtype=float)
                    messages[(node, neigh)] = Factor(
                        [node], table, {node: domain}
                    ).normalize()

        # 跟踪记录
        trace_records: List[BPIterationRecord] = []
        last_diff = 0.0
        last_iter = 0

        update_pairs: List[Tuple[Any, Any]] = []
        for node in edges:
            for neigh in edges[node]:
                if node in factor_nodes and neigh in variable_nodes:
                    update_pairs.append((node, neigh))
                elif node in variable_nodes and neigh in factor_nodes:
                    update_pairs.append((node, neigh))

        # 迭代更新消息直到收敛或达到最大迭代次数
        for iter_idx in range(1, self.max_iter + 1):
            max_diff = 0.0

            synchronous = self.update_schedule == "synchronous"
            random_order = self.update_schedule == "random"

            if synchronous:
                new_messages: Dict[Tuple[Any, Any], Factor] = dict(messages)

            pairs = list(update_pairs)
            if random_order:
                self._rng.shuffle(pairs)

            # 记录本轮更新的边（用于消息传播可视化）
            updated_edges_this_iter: List[Tuple[Any, Any]] = []

            # 更新所有消息
            for node, neigh in pairs:
                if node in factor_nodes and neigh in variable_nodes:
                    msg = self._compute_message_factor_to_var(
                        node, neigh, messages, edges, variable_nodes, factor_nodes
                    )
                elif node in variable_nodes and neigh in factor_nodes:
                    msg = self._compute_message_var_to_factor(
                        node, neigh, messages, edges, variable_nodes, factor_nodes
                    )
                else:
                    continue

                key = (node, neigh)
                old = messages.get(key)

                if old is not None and self.damping > 0.0:
                    mixed_table = (1.0 - self.damping) * np.asarray(
                        msg.table, dtype=float
                    ) + self.damping * np.asarray(old.table, dtype=float)
                    msg = Factor(
                        list(msg.scope), mixed_table, dict(msg.var_domains)
                    ).normalize()

                if old is not None:
                    diff = float(
                        np.max(
                            np.abs(
                                np.asarray(msg.table, dtype=float)
                                - np.asarray(old.table, dtype=float)
                            )
                        )
                    )
                    if diff > max_diff:
                        max_diff = diff
                    # 记录有变化的边
                    if trace_messages and diff > 1e-10:
                        updated_edges_this_iter.append(key)
                elif trace_messages:
                    # 新消息（第一次迭代）
                    updated_edges_this_iter.append(key)

                if synchronous:
                    new_messages[key] = msg
                else:
                    messages[key] = msg

            if synchronous:
                messages = new_messages

            last_iter = iter_idx
            last_diff = max_diff
            converged = max_diff < self.tol

            # 记录跟踪信息
            if trace or trace_callback or trace_beliefs or trace_messages:
                snapshot = None
                msg_snapshot = None
                
                if trace_beliefs:
                    snapshot = {
                        var: belief.table
                        for var, belief in self._collect_beliefs(
                            variable_nodes, edges, messages, evidence
                        ).items()
                    }
                
                if trace_messages:
                    # 提取变量节点间的"有效消息"（从因子到变量的消息代表约束传播）
                    # 转换为 var->var 形式以便可视化
                    msg_snapshot = self._extract_var_to_var_messages(
                        messages, factor_nodes, variable_nodes
                    )
                
                record = BPIterationRecord(
                    iteration=iter_idx,
                    residual=max_diff,
                    converged=converged,
                    beliefs=snapshot,
                    messages=msg_snapshot,
                    updated_edges=updated_edges_this_iter if trace_messages else None,
                )
                trace_records.append(record)
                if trace_callback is not None:
                    trace_callback(record)

            # 如果收敛，提前退出
            if converged:
                break

        # 保存最后一次运行的信息
        self.last_iter = last_iter
        self.last_residual = float(last_diff)
        self.last_converged = bool(last_diff < self.tol)
        if trace or trace_callback or trace_beliefs or trace_messages:
            self.last_trace = BPTrace(
                records=trace_records, max_iter=self.max_iter, tol=self.tol
            )
        else:
            self.last_trace = None

        return self._collect_beliefs(variable_nodes, edges, messages, evidence)

    def _extract_var_to_var_messages(
        self,
        messages: Dict[Tuple[Any, Any], Factor],
        factor_nodes: Dict[Any, Dict],
        variable_nodes: Dict[Any, Dict],
    ) -> Dict[Tuple[Any, Any], np.ndarray]:
        """
        从因子图消息中提取变量到变量的"有效消息"用于可视化。
        
        BP 消息实际上是在变量节点和因子节点之间传递的。为了可视化
        变量之间的信息传播，我们提取 factor->var 消息，并映射到
        原始图的边上。
        
        对于二元因子 f(u,v)，消息 f->v 代表"u 对 v 的约束/建议"。
        
        参数:
            messages: 因子图消息字典
            factor_nodes: 因子节点字典
            variable_nodes: 变量节点字典
            
        返回:
            {(src_var, dst_var): message_array} 形式的字典
        """
        var_messages: Dict[Tuple[Any, Any], np.ndarray] = {}
        
        for (src, dst), msg in messages.items():
            # 只关注 factor -> variable 的消息
            if src in factor_nodes and dst in variable_nodes:
                factor_info = factor_nodes[src]
                factor_vars = factor_info["variables"]
                
                # 对于二元因子，找到另一个变量作为"消息来源"
                other_vars = [v for v in factor_vars if v != dst]
                if len(other_vars) == 1:
                    src_var = other_vars[0]
                    var_messages[(src_var, dst)] = np.asarray(msg.table, dtype=float)
                elif len(other_vars) == 0:
                    # 一元因子（先验），标记为 None -> var
                    var_messages[(None, dst)] = np.asarray(msg.table, dtype=float)
        
        return var_messages

    def _collect_beliefs(
        self,
        variable_nodes: Dict[Any, Dict],
        edges: Dict[Any, set],
        messages: Dict[Tuple[Any, Any], Factor],
        evidence: Dict[Any, Any],
    ) -> Dict[Any, Factor]:
        """
        收集所有变量的信念（边缘概率）。
        
        参数:
            variable_nodes: 变量节点字典
            edges: 图的边
            messages: 消息字典
            evidence: 证据字典
            
        返回:
            包含所有变量信念的字典
        """
        beliefs: Dict[Any, Factor] = {}
        for var in variable_nodes.keys():
            # 跳过有证据的变量
            if var in evidence:
                continue

            # 收集所有传入该变量的消息
            incoming: List[Factor] = []
            for factor_name in edges.get(var, []):
                msg_key = (factor_name, var)
                if msg_key in messages:
                    incoming.append(messages[msg_key])

            # 如果没有传入消息，使用均匀分布
            if not incoming:
                domain = variable_nodes[var]["domain"]
                table = np.ones(len(domain), dtype=float)
                beliefs[var] = Factor([var], table, {var: domain}).normalize()
            else:
                # 将所有传入消息相乘并归一化
                beliefs[var] = Factor.multiply(incoming).normalize()

        return beliefs

    def _make_delta_factor(self, model, var: Any, value: Any) -> Factor:
        """
        创建一个Delta因子，表示变量取特定值的确定性分布。
        
        参数:
            model: 概率图模型
            var: 变量名
            value: 变量取值
            
        返回:
            表示确定性分布的因子
        """
        domain = model.var_domains[var]
        idx = domain.index(value)
        table = np.zeros(len(domain), dtype=float)
        table[idx] = 1.0
        return Factor([var], table, {var: domain})

    def _make_uniform_factor(self, model, var: Any) -> Factor:
        """
        创建一个均匀因子，表示变量的均匀分布。
        
        参数:
            model: 概率图模型
            var: 变量名
            
        返回:
            表示均匀分布的因子
        """
        domain = model.var_domains[var]
        table = np.ones(len(domain), dtype=float)
        return Factor([var], table, {var: domain}).normalize()

    def _run_bp(
        self,
        model,
        evidence: Dict[Any, Any],
        trace: bool = False,
        trace_beliefs: bool = False,
        trace_messages: bool = False,
        trace_callback: Optional[Callable[[BPIterationRecord], None]] = None,
    ) -> Dict[Any, Factor]:
        """
        运行置信传播算法。
        
        参数:
            model: 概率图模型
            evidence: 证据字典
            trace: 是否记录跟踪信息
            trace_beliefs: 是否记录每轮迭代的信念
            trace_messages: 是否记录每轮迭代的消息
            trace_callback: 可选的回调函数，每轮迭代后调用
            
        返回:
            包含所有变量信念的字典
        """
        # 处理空模型的情况
        if not hasattr(model, "get_factors") or not model.get_factors():
            self.last_iter = 0
            self.last_residual = 0.0
            self.last_converged = True
            if trace or trace_callback or trace_beliefs or trace_messages:
                record = BPIterationRecord(
                    iteration=0, residual=0.0, converged=True, beliefs=None
                )
                self.last_trace = BPTrace(records=[record], max_iter=0, tol=self.tol)
                if trace_callback is not None:
                    trace_callback(record)
            else:
                self.last_trace = None
            return {}

        # 检查图是否为树结构
        _, _, edges = self._build_factor_graph(model)
        is_tree = self._is_tree(edges)

        # 如果是树结构，使用树BP
        if is_tree:
            beliefs = self._belief_propagation_tree(model, [], evidence)
            self.last_iter = 1
            self.last_residual = 0.0
            self.last_converged = True
            if trace or trace_callback or trace_beliefs or trace_messages:
                snapshot = None
                if trace_beliefs:
                    snapshot = {var: belief.table for var, belief in beliefs.items()}
                record = BPIterationRecord(
                    iteration=1, residual=0.0, converged=True, beliefs=snapshot
                )
                self.last_trace = BPTrace(records=[record], max_iter=1, tol=self.tol)
                if trace_callback is not None:
                    trace_callback(record)
            else:
                self.last_trace = None
            return beliefs

        # 否则使用循环BP
        return self._belief_propagation_loopy(
            model,
            [],
            evidence,
            trace=trace,
            trace_beliefs=trace_beliefs,
            trace_messages=trace_messages,
            trace_callback=trace_callback,
        )

    def run_all_beliefs(
        self,
        model,
        evidence: Optional[Dict[Any, Any]] = None,
        trace: bool = False,
        trace_beliefs: bool = False,
        trace_messages: bool = False,
        trace_callback: Optional[Callable[[BPIterationRecord], None]] = None,
        include_evidence: bool = True,
        return_factors: bool = False,
    ) -> Dict[Any, Any]:
        """
        运行BP一次并返回所有变量的信念。
        
        参数:
            model: 概率图模型
            evidence: 可选的证据字典
            trace: 是否记录跟踪信息
            trace_beliefs: 是否记录每轮迭代的信念
            trace_messages: 是否记录每轮迭代的消息
            trace_callback: 可选的回调函数，每轮迭代后调用
            include_evidence: 是否在结果中包含证据变量
            return_factors: 是否返回Factor对象而不是数组
            
        返回:
            包含所有变量信念的字典
        """
        evidence = evidence or {}
        beliefs = self._run_bp(
            model,
            evidence,
            trace=trace,
            trace_beliefs=trace_beliefs,
            trace_messages=trace_messages,
            trace_callback=trace_callback,
        )

        # 如果需要，添加证据变量的信念
        if include_evidence:
            for var, value in evidence.items():
                beliefs[var] = self._make_delta_factor(model, var, value)

        # 确保所有变量都有信念
        if hasattr(model, "var_domains"):
            for var in model.var_domains:
                if var not in beliefs:
                    if var in evidence:
                        beliefs[var] = self._make_delta_factor(
                            model, var, evidence[var]
                        )
                    else:
                        beliefs[var] = self._make_uniform_factor(model, var)

        # 根据参数返回Factor对象或数组
        if return_factors:
            return beliefs

        return {var: belief.table for var, belief in beliefs.items()}

    # ------------------------------------------------------------------
    # 公共API
    # ------------------------------------------------------------------
    def query(
        self,
        model,
        query_vars: List[Any],
        evidence: Dict[Any, Any],
        elimination_order: Optional[List[Any]] = None,
    ) -> Factor:
        """
        查询变量的边缘分布。
        
        参数:
            model: 概率图模型
            query_vars: 查询变量列表
            evidence: 证据字典
            elimination_order: 消元顺序（在此实现中不使用）
            
        返回:
            查询变量的边缘分布（Factor对象）
            
        异常:
            ValueError: 如果没有查询变量或查询多个变量
        """
        if not query_vars:
            raise ValueError(
                "BeliefPropagation.query requires at least one query variable."
            )

        if len(query_vars) > 1:
            raise ValueError(
                "BeliefPropagation currently supports only single-variable marginals.\n"
                "Use VariableElimination for multi-variable joint / marginal queries."
            )

        query_var = query_vars[0]

        # 如果查询变量在证据中，返回确定性分布
        if query_var in evidence:
            self.last_iter = 0
            self.last_residual = 0.0
            self.last_converged = True
            self.last_trace = None
            return self._make_delta_factor(model, query_var, evidence[query_var])

        # 处理空模型的情况
        if not hasattr(model, "get_factors") or not model.get_factors():
            self.last_iter = 0
            self.last_residual = 0.0
            self.last_converged = True
            self.last_trace = None
            return self._make_uniform_factor(model, query_var)

        # 运行BP并获取查询变量的信念
        beliefs = self._run_bp(model, evidence)
        if query_var in beliefs:
            return beliefs[query_var].normalize()

        # 如果查询变量不在信念中，返回均匀分布
        if hasattr(model, "var_domains") and query_var in model.var_domains:
            return self._make_uniform_factor(model, query_var)

        raise ValueError(f"Query variable {query_var} not found in model.")


class InstrumentedMaxProductBeliefPropagation(InstrumentedBeliefPropagation):
    """
    最大乘积置信传播（用于MAP/最大边缘）。
    
    这是InstrumentedBeliefPropagation的子类，使用最大乘积算法
    而不是和积算法，适用于寻找最大后验概率（MAP）配置。
    """

    def _compute_message_factor_to_var(
        self,
        factor: Any,
        var: Any,
        messages: Dict[Tuple[Any, Any], Factor],
        edges: Dict[Any, set],
        variable_nodes: Dict[Any, Dict],
        factor_nodes: Dict[Any, Dict],
    ) -> Factor:
        """
        计算从因子节点到变量节点的消息（最大乘积版本）。
        
        与父类的主要区别是使用max_out而不是sum_out来边缘化变量。
        
        参数:
            factor: 因子节点名称
            var: 变量节点名称
            messages: 当前消息字典
            edges: 图的边
            variable_nodes: 变量节点字典
            factor_nodes: 因子节点字典
            
        返回:
            计算出的消息（Factor对象）
        """
        factor_obj: Factor = factor_nodes[factor]["factor"]
        other_vars = [v for v in factor_obj.scope if v != var]

        if not other_vars:
            return factor_obj.copy().normalize()

        # 收集因子和来自其他变量的消息
        message_factors: List[Factor] = [factor_obj]
        for other_var in other_vars:
            msg_key = (other_var, factor)
            if msg_key in messages:
                message_factors.append(messages[msg_key])

        # 将所有因子和消息相乘
        combined = Factor.multiply(message_factors)

        # 对除目标变量外的所有变量求最大值（最大边缘化）
        for other_var in other_vars:
            if other_var in combined.scope:
                combined = combined.max_out(other_var)

        return combined.normalize()
