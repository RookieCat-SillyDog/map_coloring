"""
模块 2：日志结构（SearchLogger）
从当前 steps 列表升级为结构化日志
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from collections import defaultdict


@dataclass
class SearchStepRecord:
    """Step 级日志结构"""
    step_id: int
    action: str  # "try", "assign", "reject", "backtrack", "success"
    var: Optional[str]  # region 名称，success 时可为 None
    color: Optional[str]  # 颜色或特殊标记（如 "ALL_TRIED"）
    depth: int  # 当前 assignment 的深度（赋值变量数）
    assignment_snapshot: Optional[dict[str, str]]  # 当前赋值的浅拷贝
    conflict_neighbor: Optional[str]  # 若 action=="reject"，导致冲突的邻居
    stack_size: int  # 当前栈大小
    remaining_colors_for_var: Optional[int]  # 当前 var 剩余未尝试颜色数


@dataclass
class BacktrackEvent:
    """回溯事件日志"""
    event_id: int
    step_id: int  # 触发 backtrack 的 step_id
    depth_before: int  # 回溯前深度
    depth_after: int  # 回溯后深度
    backtrack_depth: int  # depth_before - depth_after


@dataclass
class VariableStats:
    """变量级统计"""
    assign_count: int = 0
    reject_count: int = 0
    try_count: int = 0
    color_try_counts: dict[str, int] = field(default_factory=dict)
    max_depth_assigned: int = 0  # 该变量被成功赋值时达到的最大深度


@dataclass
class EdgeStats:
    """约束边级统计"""
    conflict_count: int = 0
    conflict_depths: list[int] = field(default_factory=list)


@dataclass
class SearchRunSummary:
    """运行级汇总"""
    total_steps: int = 0
    total_assigns: int = 0
    total_rejects: int = 0
    total_backtracks: int = 0
    max_depth_reached: int = 0
    max_stack_size: int = 0
    solved: bool = False
    solution: Optional[dict[str, str]] = None
    solutions_found: int = 0  # 找到的解的数量（find_all_solutions 模式）


class SearchLogger:
    """搜索日志记录器"""
    
    def __init__(self, regions: list[str], log_snapshots: bool = True):
        """
        初始化日志记录器
        
        Args:
            regions: 变量（区域）列表
            log_snapshots: 是否记录赋值快照
        """
        self.regions = regions
        self.log_snapshots = log_snapshots
        
        # Step 级日志
        self.steps: list[SearchStepRecord] = []
        
        # 回溯事件日志
        self.backtrack_events: list[BacktrackEvent] = []
        
        # 变量级统计
        self.variable_stats: dict[str, VariableStats] = {
            r: VariableStats() for r in regions
        }
        
        # 边级统计（使用 frozenset 作为无向边的 key）
        self._edge_stats: dict[frozenset, EdgeStats] = defaultdict(EdgeStats)
        
        # 运行级汇总
        self.summary = SearchRunSummary()
        
        # 内部状态
        self._step_counter = 0
        self._backtrack_counter = 0
        self._last_depth = 0
        self._max_stack_size = 0
    
    def _get_edge_key(self, u: str, v: str) -> frozenset:
        """获取无向边的 key"""
        return frozenset([u, v])
    
    def log_step(
        self,
        action: str,
        var: Optional[str],
        color: Optional[str],
        depth: int,
        assignment: Optional[dict[str, str]],
        stack_size: int,
        conflict_neighbor: Optional[str] = None,
        remaining_colors: Optional[int] = None,
    ) -> SearchStepRecord:
        """
        记录一个搜索步骤
        
        Args:
            action: 动作类型 ("try", "assign", "reject", "backtrack", "success")
            var: 当前变量
            color: 当前颜色
            depth: 当前深度
            assignment: 当前赋值
            stack_size: 栈大小
            conflict_neighbor: 冲突邻居（reject 时）
            remaining_colors: 剩余颜色数
        
        Returns:
            创建的日志记录
        """
        self._step_counter += 1
        
        # 更新最大栈大小
        if stack_size > self._max_stack_size:
            self._max_stack_size = stack_size
        
        # 创建记录
        record = SearchStepRecord(
            step_id=self._step_counter,
            action=action,
            var=var,
            color=color,
            depth=depth,
            assignment_snapshot=assignment.copy() if (assignment and self.log_snapshots) else None,
            conflict_neighbor=conflict_neighbor,
            stack_size=stack_size,
            remaining_colors_for_var=remaining_colors,
        )
        self.steps.append(record)
        
        # 更新变量级统计
        if var and var in self.variable_stats:
            stats = self.variable_stats[var]
            
            if action == "try":
                stats.try_count += 1
                if color:
                    stats.color_try_counts[color] = stats.color_try_counts.get(color, 0) + 1
            
            elif action == "assign":
                stats.assign_count += 1
                if depth > stats.max_depth_assigned:
                    stats.max_depth_assigned = depth
            
            elif action == "reject":
                stats.reject_count += 1
                # 记录边级冲突
                if conflict_neighbor:
                    edge_key = self._get_edge_key(var, conflict_neighbor)
                    edge_stats = self._edge_stats[edge_key]
                    edge_stats.conflict_count += 1
                    edge_stats.conflict_depths.append(depth)
        
        # 更新汇总统计
        self.summary.total_steps += 1
        if action == "assign":
            self.summary.total_assigns += 1
        elif action == "reject":
            self.summary.total_rejects += 1
        
        if depth > self.summary.max_depth_reached:
            self.summary.max_depth_reached = depth
        
        if stack_size > self.summary.max_stack_size:
            self.summary.max_stack_size = stack_size
        
        self._last_depth = depth
        
        return record
    
    def log_backtrack(self, step_id: int, depth_before: int, depth_after: int) -> BacktrackEvent:
        """
        记录回溯事件
        
        Args:
            step_id: 触发回溯的步骤 ID
            depth_before: 回溯前深度
            depth_after: 回溯后深度
        
        Returns:
            创建的回溯事件
        """
        self._backtrack_counter += 1
        
        event = BacktrackEvent(
            event_id=self._backtrack_counter,
            step_id=step_id,
            depth_before=depth_before,
            depth_after=depth_after,
            backtrack_depth=depth_before - depth_after,
        )
        self.backtrack_events.append(event)
        
        self.summary.total_backtracks += 1
        
        return event

    
    def finalize_run(
        self,
        solved: bool,
        solution: Optional[dict[str, str]] = None,
        solutions_found: int = 0,
    ):
        """
        完成搜索运行，生成最终汇总
        
        Args:
            solved: 是否找到解
            solution: 找到的解（如果有）
            solutions_found: 找到的解的数量
        """
        self.summary.solved = solved
        self.summary.solution = solution.copy() if solution else None
        self.summary.solutions_found = max(solutions_found, 1 if solved else 0)
        self.summary.max_stack_size = self._max_stack_size
    
    def get_edge_stats(self, u: str, v: str) -> EdgeStats:
        """获取指定边的统计"""
        return self._edge_stats[self._get_edge_key(u, v)]
    
    def get_all_edge_stats(self) -> dict[tuple[str, str], EdgeStats]:
        """获取所有边的统计（返回元组 key 便于使用）"""
        result = {}
        for edge_key, stats in self._edge_stats.items():
            edge_list = sorted(list(edge_key))
            if len(edge_list) == 2:
                result[(edge_list[0], edge_list[1])] = stats
        return result
    
    def get_steps_by_action(self, action: str) -> list[SearchStepRecord]:
        """获取指定动作类型的所有步骤"""
        return [s for s in self.steps if s.action == action]
    
    def get_steps_for_var(self, var: str) -> list[SearchStepRecord]:
        """获取指定变量的所有步骤"""
        return [s for s in self.steps if s.var == var]
    
    def print_summary(self):
        """打印搜索汇总"""
        s = self.summary
        print("=" * 50)
        print("搜索运行汇总")
        print("=" * 50)
        print(f"总步数: {s.total_steps}")
        print(f"总赋值次数: {s.total_assigns}")
        print(f"总拒绝次数: {s.total_rejects}")
        print(f"总回溯次数: {s.total_backtracks}")
        print(f"最大深度: {s.max_depth_reached}")
        print(f"最大栈大小: {s.max_stack_size}")
        print(f"是否求解成功: {'是' if s.solved else '否'}")
        if s.solution:
            print(f"解: {s.solution}")
        print("=" * 50)
    
    def print_steps(self, max_steps: int = 30):
        """打印搜索步骤"""
        action_symbols = {
            "try": "?",
            "reject": "✗",
            "assign": "✓",
            "backtrack": "←",
            "success": "★",
        }
        
        print(f"\n搜索过程（前 {min(max_steps, len(self.steps))} 步）：")
        print("-" * 60)
        
        for step in self.steps[:max_steps]:
            symbol = action_symbols.get(step.action, " ")
            if step.action == "success":
                print(f"[{step.step_id:3}] {symbol} 找到解!")
            elif step.action == "backtrack":
                print(f"[{step.step_id:3}] {symbol} {step.var}: 所有颜色已尝试，回溯")
            else:
                conflict_info = f" (与 {step.conflict_neighbor} 冲突)" if step.conflict_neighbor else ""
                print(f"[{step.step_id:3}] {symbol} {step.var}: {step.color} ({step.action}){conflict_info} [深度:{step.depth}]")
        
        if len(self.steps) > max_steps:
            print(f"... 省略 {len(self.steps) - max_steps} 步 ...")
