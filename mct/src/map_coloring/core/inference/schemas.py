# -*- coding: utf-8 -*-
"""
推理过程追踪和结果标准化协议。

本模块定义了用于记录推理算法执行过程和结果的数据结构：
- TraceStep: 单个推理步骤的记录
- TraceSchema: 推理过程追踪协议
- ResultSchema: 推理结果标准化结构
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class TraceStep:
    """单个推理步骤的记录。
    
    Attributes:
        step_id: 步骤的唯一标识符（从0开始递增）
        action: 动作类型，如 'assign', 'backtrack', 'try', 'reject' 等
        timestamp: 步骤执行的时间戳（Unix时间）
        depth: 当前搜索深度
        var: 相关变量名（可选）
        value: 相关值（可选）
        metadata: 额外的元数据字典
    """
    step_id: int
    action: str
    timestamp: float
    depth: int
    var: Optional[str] = None
    value: Optional[Any] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TraceSchema:
    """推理过程追踪协议。
    
    用于记录算法执行过程中的所有步骤，支持追踪和分析。
    
    Attributes:
        algorithm: 算法名称
        start_time: 开始执行的时间
        steps: 执行步骤列表
    
    Example:
        >>> trace = TraceSchema(algorithm="CSP-Backtracking", start_time=datetime.now())
        >>> trace.add_step(action="assign", depth=0, var="A", value="red")
        >>> trace.add_step(action="backtrack", depth=0)
        >>> len(trace.steps)
        2
    """
    algorithm: str
    start_time: datetime
    steps: list[TraceStep] = field(default_factory=list)
    
    def add_step(self, action: str, depth: int, **kwargs) -> TraceStep:
        """添加一个新的追踪步骤。
        
        Args:
            action: 动作类型
            depth: 当前搜索深度
            **kwargs: 传递给 TraceStep 的额外参数（var, value, metadata）
        
        Returns:
            新创建的 TraceStep 实例
        """
        step = TraceStep(
            step_id=len(self.steps),
            action=action,
            timestamp=time.time(),
            depth=depth,
            **kwargs
        )
        self.steps.append(step)
        return step


@dataclass
class ResultSchema:
    """推理结果标准化结构。
    
    用于统一不同算法的输出格式，便于比较和分析。
    
    Attributes:
        algorithm: 算法名称
        solved: 是否成功求解
        solution: 求解结果（如变量赋值字典）
        elapsed_seconds: 执行耗时（秒）
        total_steps: 总步骤数
        metadata: 额外的元数据字典
    
    Example:
        >>> result = ResultSchema(
        ...     algorithm="CSP-Backtracking",
        ...     solved=True,
        ...     solution={"A": "red", "B": "blue", "C": "green"},
        ...     elapsed_seconds=0.05,
        ...     total_steps=10
        ... )
        >>> result.solved
        True
    """
    algorithm: str
    solved: bool
    solution: Optional[dict] = None
    elapsed_seconds: float = 0.0
    total_steps: int = 0
    metadata: dict = field(default_factory=dict)
