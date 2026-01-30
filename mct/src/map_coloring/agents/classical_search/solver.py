"""
模块 3：核心搜索循环重构
在保持当前 DFS 语义的基础上引入策略与日志
"""

import os
import sys
from typing import Optional
from copy import deepcopy

from map_coloring.agents.classical_search.config import CSPConfig
from map_coloring.agents.classical_search.logger import SearchLogger


class Stack:
    """简单栈实现（避免外部依赖）"""
    def __init__(self):
        self._items = []
    
    def push(self, item):
        self._items.append(item)
    
    def pop(self):
        return self._items.pop()
    
    def is_empty(self) -> bool:
        return len(self._items) == 0
    
    def size(self) -> int:
        return len(self._items)


class CSPSolver:
    """可配置的 CSP 求解器"""
    
    def __init__(
        self,
        regions: list[str],
        neighbors: dict[str, list[str]],
        colors: list[str],
        config: Optional[CSPConfig] = None,
    ):
        """
        初始化求解器
        
        Args:
            regions: 变量（区域）列表
            neighbors: 邻接关系字典
            colors: 可用颜色列表
            config: 求解器配置
        """
        self.regions = regions
        self.neighbors = neighbors
        self.colors = colors
        self.config = config or CSPConfig()
        
        # 验证配置
        errors = self.config.validate()
        if errors:
            raise ValueError(f"配置错误: {errors}")
        
        # 初始化日志记录器
        self.logger = SearchLogger(
            regions=regions,
            log_snapshots=self.config.log_assignment_snapshots,
        )
        
        # 搜索结果
        self.solution: Optional[dict[str, str]] = None
        self.all_solutions: list[dict[str, str]] = []
    
    def select_next_var(
        self,
        assignment: dict[str, str],
        domains_state: Optional[dict[str, set[str]]],
    ) -> Optional[str]:
        """
        选择下一个要处理的变量
        
        Args:
            assignment: 当前赋值
            domains_state: 域状态（forward checking 时使用）
        
        Returns:
            下一个变量名，或 None（所有变量已赋值）
        """
        # 获取未赋值变量
        unassigned = [r for r in self.regions if r not in assignment]
        
        if not unassigned:
            return None
        
        strategy = self.config.var_ordering_strategy
        
        if strategy == "static":
            # 按原始顺序
            return unassigned[0]
        
        elif strategy == "random":
            # 随机选择
            return self.config.rng.choice(unassigned)
        
        elif strategy == "mrv":
            # 最小剩余值
            return self._select_mrv(unassigned, domains_state)
        
        elif strategy == "degree":
            # 度启发式
            return self._select_degree(unassigned, assignment)
        
        elif strategy == "mrv+degree":
            # MRV + 度启发式
            return self._select_mrv_degree(unassigned, assignment, domains_state)
        
        else:
            return unassigned[0]
    
    def _select_mrv(
        self,
        unassigned: list[str],
        domains_state: Optional[dict[str, set[str]]],
    ) -> str:
        """MRV 启发式：选择剩余合法值最少的变量"""
        if domains_state is None:
            # 没有域状态，退化为静态
            return unassigned[0]
        
        min_remaining = float('inf')
        candidates = []
        
        for var in unassigned:
            remaining = len(domains_state.get(var, self.colors))
            if remaining < min_remaining:
                min_remaining = remaining
                candidates = [var]
            elif remaining == min_remaining:
                candidates.append(var)
        
        if self.config.random_tie_breaking and len(candidates) > 1:
            return self.config.rng.choice(candidates)
        return candidates[0]
    
    def _select_degree(
        self,
        unassigned: list[str],
        assignment: dict[str, str],
    ) -> str:
        """度启发式：选择与未赋值变量约束最多的变量"""
        max_degree = -1
        candidates = []
        
        for var in unassigned:
            # 计算与未赋值邻居的约束数
            degree = sum(1 for nb in self.neighbors[var] if nb not in assignment)
            if degree > max_degree:
                max_degree = degree
                candidates = [var]
            elif degree == max_degree:
                candidates.append(var)
        
        if self.config.random_tie_breaking and len(candidates) > 1:
            return self.config.rng.choice(candidates)
        return candidates[0]
    
    def _select_mrv_degree(
        self,
        unassigned: list[str],
        assignment: dict[str, str],
        domains_state: Optional[dict[str, set[str]]],
    ) -> str:
        """MRV + 度启发式组合"""
        if domains_state is None:
            return self._select_degree(unassigned, assignment)
        
        # 先按 MRV 筛选
        min_remaining = float('inf')
        mrv_candidates = []
        
        for var in unassigned:
            remaining = len(domains_state.get(var, self.colors))
            if remaining < min_remaining:
                min_remaining = remaining
                mrv_candidates = [var]
            elif remaining == min_remaining:
                mrv_candidates.append(var)
        
        if len(mrv_candidates) == 1:
            return mrv_candidates[0]
        
        # 再按度启发式选择
        return self._select_degree(mrv_candidates, assignment)
    
    def order_colors_for_var(
        self,
        var: str,
        assignment: dict[str, str],
        domains_state: Optional[dict[str, set[str]]],
    ) -> list[str]:
        """
        为变量排序颜色
        
        Args:
            var: 变量名
            assignment: 当前赋值
            domains_state: 域状态
        
        Returns:
            排序后的颜色列表
        """
        # 获取可用颜色
        if domains_state and var in domains_state:
            available_colors = list(domains_state[var])
        else:
            available_colors = list(self.colors)
        
        strategy = self.config.value_ordering_strategy
        
        if strategy == "static":
            return available_colors
        
        elif strategy == "random":
            shuffled = available_colors.copy()
            self.config.rng.shuffle(shuffled)
            return shuffled
        
        elif strategy == "least_constraining":
            return self._order_least_constraining(var, available_colors, assignment, domains_state)
        
        else:
            return available_colors

    
    def _order_least_constraining(
        self,
        var: str,
        colors: list[str],
        assignment: dict[str, str],
        domains_state: Optional[dict[str, set[str]]],
    ) -> list[str]:
        """最少约束值：选择对邻居限制最少的值"""
        if domains_state is None:
            return colors
        
        def count_constraints(color: str) -> int:
            """计算选择该颜色会排除多少邻居的选择"""
            count = 0
            for nb in self.neighbors[var]:
                if nb not in assignment and nb in domains_state:
                    if color in domains_state[nb]:
                        count += 1
            return count
        
        # 按约束数升序排序
        return sorted(colors, key=count_constraints)
    
    def _init_domains(self) -> dict[str, set[str]]:
        """初始化所有变量的域"""
        return {var: set(self.colors) for var in self.regions}
    
    def _forward_check(
        self,
        var: str,
        color: str,
        assignment: dict[str, str],
        domains_state: dict[str, set[str]],
    ) -> tuple[bool, dict[str, set[str]]]:
        """
        前向检查：赋值后更新邻居的域
        
        Args:
            var: 被赋值的变量
            color: 赋的值
            assignment: 当前赋值
            domains_state: 当前域状态
        
        Returns:
            (是否有效, 新的域状态)
        """
        # 深拷贝域状态
        new_domains = {k: v.copy() for k, v in domains_state.items()}
        
        # 从邻居的域中移除当前颜色
        for nb in self.neighbors[var]:
            if nb not in assignment and nb in new_domains:
                new_domains[nb].discard(color)
                # 如果某个邻居的域为空，则失败
                if len(new_domains[nb]) == 0:
                    return False, domains_state
        
        return True, new_domains
    
    def _is_consistent(
        self,
        var: str,
        color: str,
        assignment: dict[str, str],
    ) -> tuple[bool, Optional[str]]:
        """
        检查赋值是否一致
        
        Args:
            var: 变量
            color: 颜色
            assignment: 当前赋值
        
        Returns:
            (是否一致, 冲突邻居名或 None)
        """
        for nb in self.neighbors[var]:
            if assignment.get(nb) == color:
                return False, nb
        return True, None
    
    def solve(self) -> Optional[dict[str, str]]:
        """
        执行搜索求解
        
        Returns:
            找到的解，或 None
        """
        # 重置状态
        self.solution = None
        self.all_solutions = []
        self.logger = SearchLogger(
            regions=self.regions,
            log_snapshots=self.config.log_assignment_snapshots,
        )
        self.config.reset_rng()
        
        # 初始化栈
        stack = Stack()
        
        # 初始化域状态（如果启用 forward checking）
        initial_domains = self._init_domains() if self.config.use_forward_checking else None
        
        # 初始状态：空赋值，选择第一个变量，颜色列表
        first_var = self.select_next_var({}, initial_domains)
        if first_var is None:
            # 没有变量需要赋值
            self.logger.finalize_run(True, {})
            return {}
        
        initial_colors = self.order_colors_for_var(first_var, {}, initial_domains)
        
        # 栈状态格式：(assignment, var, color_list, color_idx, domains_state)
        stack.push(({}, first_var, initial_colors, 0, initial_domains))
        
        # 统计
        backtrack_count = 0
        last_depth = 0
        
        # 主搜索循环
        while not stack.is_empty():
            # 检查资源限制
            if self.config.max_steps and self.logger.summary.total_steps >= self.config.max_steps:
                break
            if self.config.max_backtracks and backtrack_count >= self.config.max_backtracks:
                break
            
            # 弹出当前状态
            assignment, var, color_list, color_idx, domains_state = stack.pop()
            current_depth = len(assignment)
            
            # 检查是否需要记录回溯
            if current_depth < last_depth:
                self.logger.log_backtrack(
                    step_id=self.logger._step_counter,
                    depth_before=last_depth,
                    depth_after=current_depth,
                )
                backtrack_count += 1
            
            # 检查是否所有颜色都试完
            if color_idx >= len(color_list):
                # 记录回溯步骤
                self.logger.log_step(
                    action="backtrack",
                    var=var,
                    color="ALL_TRIED",
                    depth=current_depth,
                    assignment=assignment,
                    stack_size=stack.size(),
                    remaining_colors=0,
                )
                last_depth = current_depth
                continue
            
            # 当前尝试的颜色
            color = color_list[color_idx]
            remaining_colors = len(color_list) - color_idx
            
            # 记录尝试
            self.logger.log_step(
                action="try",
                var=var,
                color=color,
                depth=current_depth,
                assignment=assignment,
                stack_size=stack.size(),
                remaining_colors=remaining_colors,
            )
            
            # 一致性检查
            is_consistent, conflict_neighbor = self._is_consistent(var, color, assignment)
            
            if not is_consistent:
                # 冲突，记录拒绝
                self.logger.log_step(
                    action="reject",
                    var=var,
                    color=color,
                    depth=current_depth,
                    assignment=assignment,
                    stack_size=stack.size(),
                    conflict_neighbor=conflict_neighbor,
                    remaining_colors=remaining_colors - 1,
                )
                # 尝试下一个颜色
                stack.push((assignment, var, color_list, color_idx + 1, domains_state))
                last_depth = current_depth
                continue
            
            # Forward checking
            new_domains = domains_state
            if self.config.use_forward_checking and domains_state is not None:
                fc_valid, new_domains = self._forward_check(var, color, assignment, domains_state)
                if not fc_valid:
                    # Forward checking 失败
                    self.logger.log_step(
                        action="reject",
                        var=var,
                        color=color,
                        depth=current_depth,
                        assignment=assignment,
                        stack_size=stack.size(),
                        conflict_neighbor="FC_FAIL",
                        remaining_colors=remaining_colors - 1,
                    )
                    stack.push((assignment, var, color_list, color_idx + 1, domains_state))
                    last_depth = current_depth
                    continue
            
            # 赋值成功
            new_assignment = assignment.copy()
            new_assignment[var] = color
            new_depth = len(new_assignment)
            
            self.logger.log_step(
                action="assign",
                var=var,
                color=color,
                depth=new_depth,
                assignment=new_assignment,
                stack_size=stack.size(),
                remaining_colors=remaining_colors - 1,
            )
            
            # 检查是否找到解
            if new_depth == len(self.regions):
                # 找到解
                self.logger.log_step(
                    action="success",
                    var=None,
                    color=None,
                    depth=new_depth,
                    assignment=new_assignment,
                    stack_size=stack.size(),
                )
                
                if self.config.find_all_solutions:
                    self.all_solutions.append(new_assignment.copy())
                    # 继续搜索其他解：尝试当前变量的下一个颜色
                    stack.push((assignment, var, color_list, color_idx + 1, domains_state))
                else:
                    self.solution = new_assignment.copy()
                    self.logger.finalize_run(True, self.solution)
                    return self.solution
                
                last_depth = new_depth
                continue
            
            # 选择下一个变量
            next_var = self.select_next_var(new_assignment, new_domains)
            if next_var is None:
                # 不应该发生
                last_depth = new_depth
                continue
            
            next_colors = self.order_colors_for_var(next_var, new_assignment, new_domains)
            
            # 双重压栈
            # 1. 回溯点：当前变量的下一个颜色
            stack.push((assignment, var, color_list, color_idx + 1, domains_state))
            # 2. 前进点：下一个变量
            stack.push((new_assignment, next_var, next_colors, 0, new_domains))
            
            last_depth = new_depth
        
        # 搜索结束
        if self.config.find_all_solutions and self.all_solutions:
            self.solution = self.all_solutions[0]
            self.logger.finalize_run(True, self.solution, len(self.all_solutions))
            return self.solution
        
        self.logger.finalize_run(False)
        return None
