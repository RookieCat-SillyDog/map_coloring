# -*- coding: utf-8 -*-
"""
推理协议层。

本模块导出推理相关的核心类：
- TraceStep: 单个推理步骤的记录
- TraceSchema: 推理过程追踪协议
- ResultSchema: 推理结果标准化结构
- InferenceAdapter: 推理算法适配器抽象基类
"""

from map_coloring.core.inference.schemas import TraceStep, TraceSchema, ResultSchema
from map_coloring.core.inference.adapters import InferenceAdapter

__all__ = [
    "TraceStep",
    "TraceSchema",
    "ResultSchema",
    "InferenceAdapter",
]
