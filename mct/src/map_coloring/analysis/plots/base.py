# -*- coding: utf-8 -*-
"""
通用绘图工具：颜色、布局、图例等基础设施
"""

from typing import List, Optional, Tuple
import numpy as np

# 默认调色板
DEFAULT_COLORS = [
    '#FF6B6B',  # 红
    '#4ECDC4',  # 青
    '#45B7D1',  # 蓝
    '#96CEB4',  # 绿
    '#FFEAA7',  # 黄
    '#DDA0DD',  # 紫
    '#F39C12',  # 橙
    '#85C1E9',  # 浅蓝
]


def get_color(index: int, palette: Optional[List[str]] = None) -> str:
    """根据索引获取颜色"""
    colors = palette or DEFAULT_COLORS
    return colors[index % len(colors)]


def safe_normalize(probs: np.ndarray) -> np.ndarray:
    """安全归一化概率分布"""
    probs = np.asarray(probs, dtype=float)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(probs, 1.0 / len(probs), dtype=float)
    return probs / total


def compute_entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    """计算熵"""
    p = safe_normalize(probs)
    return float(-np.sum(p * np.log(p + eps)))


def format_prob_array(probs: np.ndarray, precision: int = 2) -> str:
    """格式化概率数组为字符串"""
    return "[" + ",".join([f"{p:.{precision}f}" for p in probs]) + "]"
