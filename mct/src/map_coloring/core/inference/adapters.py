# -*- coding: utf-8 -*-
"""
推理算法适配器抽象基类。

本模块定义了用于集成第三方推理库的适配器接口：
- InferenceAdapter: 推理算法适配器抽象基类

适配器模式允许将不同的推理算法（如 CSP 求解器、BP 算法等）
统一到相同的接口下，便于在实验中进行比较和切换。
"""

from abc import ABC, abstractmethod
from typing import Any

from map_coloring.core.inference.schemas import TraceSchema, ResultSchema


class InferenceAdapter(ABC):
    """推理算法适配器抽象基类。
    
    所有推理算法的适配器都应继承此类，并实现 solve() 和 get_trace() 方法。
    这确保了不同算法可以通过统一的接口进行调用和比较。
    
    Example:
        >>> class MyCSPAdapter(InferenceAdapter):
        ...     def __init__(self, config):
        ...         self._trace = None
        ...         self._config = config
        ...     
        ...     def solve(self, problem):
        ...         # 执行求解逻辑
        ...         return ResultSchema(algorithm="MyCSP", solved=True, ...)
        ...     
        ...     def get_trace(self):
        ...         return self._trace
    
    Notes:
        - solve() 方法应返回标准化的 ResultSchema 结果
        - get_trace() 方法应返回记录执行过程的 TraceSchema
        - 子类可以在构造函数中接受特定的配置参数
    """
    
    @abstractmethod
    def solve(self, problem: Any) -> ResultSchema:
        """执行推理并返回标准化结果。
        
        Args:
            problem: 待求解的问题实例。具体类型取决于算法实现，
                     例如 CSP 问题可能是包含变量、域和约束的字典。
        
        Returns:
            ResultSchema: 包含求解结果的标准化数据结构，
                         包括是否成功、解、耗时等信息。
        
        Raises:
            NotImplementedError: 子类必须实现此方法。
        """
        pass
    
    @abstractmethod
    def get_trace(self) -> TraceSchema:
        """获取推理过程追踪。
        
        返回最近一次 solve() 调用的执行追踪记录。
        如果尚未调用 solve()，行为由子类定义（可能返回空追踪或抛出异常）。
        
        Returns:
            TraceSchema: 包含执行步骤记录的追踪数据结构。
        
        Raises:
            NotImplementedError: 子类必须实现此方法。
        """
        pass
