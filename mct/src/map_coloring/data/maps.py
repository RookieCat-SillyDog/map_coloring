"""
预定义地图配置库
"""

import math
from typing import TypedDict, Optional


class MapConfig(TypedDict, total=False):
    """地图配置类型"""
    regions: list[str]
    neighbors: dict[str, list[str]]
    colors: list[str]
    visual: str
    layout: Optional[dict[str, tuple[float, float]]]  # 可选的节点布局


def _pentagon_layout() -> dict[str, tuple[float, float]]:
    """正五边形 + 中心点布局"""
    radius = 1.0
    # A在顶部，顺时针 A->B->C->D->E
    angles = [90, 18, -54, -126, -198]
    nodes = ['A', 'B', 'C', 'D', 'E']
    pos = {}
    for i, node in enumerate(nodes):
        angle_rad = math.radians(angles[i])
        pos[node] = (radius * math.cos(angle_rad), radius * math.sin(angle_rad))
    pos['O'] = (0.0, 0.0)
    return pos


def _triangle_layout() -> dict[str, tuple[float, float]]:
    """等边三角形布局"""
    return {
        'A': (0.0, 1.0),
        'B': (-math.sqrt(3)/2, -0.5),
        'C': (math.sqrt(3)/2, -0.5),
    }


def _square_layout() -> dict[str, tuple[float, float]]:
    """正方形布局"""
    return {
        'A': (-0.7, 0.7),
        'B': (0.7, 0.7),
        'C': (0.7, -0.7),
        'D': (-0.7, -0.7),
    }


def _australia_layout() -> dict[str, tuple[float, float]]:
    """澳大利亚地图布局（大致地理位置）"""
    return {
        'WA': (-1.2, 0.3),
        'NT': (-0.3, 0.8),
        'SA': (-0.2, -0.2),
        'Q': (0.6, 0.6),
        'NSW': (0.7, -0.1),
        'V': (0.5, -0.6),
        'T': (0.7, -1.1),
    }


def _corridor_layout() -> dict[str, tuple[float, float]]:
    """走廊地图布局：主干 C1-C2-C3 横向排列，侧房在上下"""
    return {
        'L1': (-1.0, 0.8),   # C1 的侧房（上方）
        'C1': (-1.0, 0.0),   # 左端
        'C2': (0.0, 0.0),    # 中间
        'R1': (0.0, -0.8),   # C2 的侧房（下方）
        'C3': (1.0, 0.0),    # 右端
        'L2': (1.0, 0.8),    # C3 的侧房（上方）
    }


def _gala_layout() -> dict[str, tuple[float, float]]:
    """gala 地图布局：主干 C1-C2-C3，C1 上挂 L1，C2 下挂 M2-F2"""
    return {
        'L1': (-1.0, 0.8),   # C1 的侧房（上方）
        'C1': (-1.0, 0.0),   # 左端
        'C2': (0.0, 0.0),    # 中间
        'M2': (0.0, -0.8),   # C2 下方的中间节点
        'F2': (0.0, -1.6),   # 分支末端
        'C3': (1.0, 0.0),    # 右端
    }


MAPS: dict[str, MapConfig] = {
    # 澳大利亚地图（经典 CSP 问题）
    "australia": {
        "regions": ["WA", "NT", "SA", "Q", "NSW", "V", "T"],
        "neighbors": {
            "WA": ["NT", "SA"],
            "NT": ["WA", "SA", "Q"],
            "SA": ["WA", "NT", "Q", "NSW", "V"],
            "Q": ["NT", "SA", "NSW"],
            "NSW": ["SA", "Q", "V"],
            "V": ["SA", "NSW", "T"],
            "T": ["V"],
        },
        "colors": ["红", "绿", "蓝"],
        "layout": _australia_layout(),
    },
    
    # 简单三角形地图（3 个区域）
    "triangle": {
        "regions": ["A", "B", "C"],
        "neighbors": {
            "A": ["B", "C"],
            "B": ["A", "C"],
            "C": ["A", "B"],
        },
        "colors": ["红", "绿", "蓝"],
        "layout": _triangle_layout(),
    },
    
    # 四方形地图（4 个区域）
    "square": {
        "regions": ["A", "B", "C", "D"],
        "neighbors": {
            "A": ["B", "D"],
            "B": ["A", "C"],
            "C": ["B", "D"],
            "D": ["A", "C"],
        },
        "colors": ["红", "绿"],
        "layout": _square_layout(),
    },
    
    # 完全图 K4（4 个节点全连接）
    "k4": {
        "regions": ["A", "B", "C", "D"],
        "neighbors": {
            "A": ["B", "C", "D"],
            "B": ["A", "C", "D"],
            "C": ["A", "B", "D"],
            "D": ["A", "B", "C"],
        },
        "colors": ["红", "绿", "蓝", "黄"],
        "layout": _square_layout(),
    },
    
    # 五边形 + 中心区域（pentagon）
    "pentagon": {
        "regions": ["O", "A", "B", "C", "D", "E"],
        "neighbors": {
            "A": ["B", "E", "O"],
            "B": ["A", "C", "O"],
            "C": ["B", "D", "O"],
            "D": ["C", "E", "O"],
            "E": ["D", "A", "O"],
            "O": ["A", "B", "C", "D", "E"],
        },
        "colors": ["红", "绿", "蓝", "黄"],
        "layout": _pentagon_layout(),
    },
    
    # 彼得森图（Petersen Graph）- 经典难图
    "petersen": {
        "regions": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "neighbors": {
            "0": ["1", "4", "5"],
            "1": ["0", "2", "6"],
            "2": ["1", "3", "7"],
            "3": ["2", "4", "8"],
            "4": ["0", "3", "9"],
            "5": ["0", "7", "8"],
            "6": ["1", "8", "9"],
            "7": ["2", "5", "9"],
            "8": ["3", "5", "6"],
            "9": ["4", "6", "7"],
        },
        "colors": ["红", "绿", "蓝"],
        # 无预定义布局，使用自动布局
    },
    
    # 走廊地图（6 区域，树状结构）
    # C1-C2-C3 构成主干，L1/R1/L2 是侧房
    "corridor": {
        "regions": ["C1", "C2", "C3", "L1", "R1", "L2"],
        "neighbors": {
            "C1": ["C2", "L1"],
            "C2": ["C1", "C3", "R1"],
            "C3": ["C2", "L2"],
            "L1": ["C1"],
            "R1": ["C2"],
            "L2": ["C3"],
        },
        "colors": ["红", "绿"],  # 树状结构只需 2 色
        "layout": _corridor_layout(),
    },
    
    # gala 地图（6 区域，带长分支的树）
    # C1-C2-C3 主干，L1 挂 C1，M2-F2 挂 C2
    "gala": {
        "regions": ["C1", "C2", "C3", "L1", "M2", "F2"],
        "neighbors": {
            "C1": ["C2", "L1"],
            "C2": ["C1", "C3", "M2"],
            "C3": ["C2"],
            "L1": ["C1"],
            "M2": ["C2", "F2"],
            "F2": ["M2"],
        },
        "colors": ["红", "绿"],  # 树状结构只需 2 色
        "layout": _gala_layout(),
    },
}


def get_map(name: str) -> MapConfig:
    """获取地图配置"""
    if name not in MAPS:
        raise ValueError(f"未知地图: {name}. 可用地图: {list(MAPS.keys())}")
    return MAPS[name]


def list_maps() -> list[str]:
    """列出所有可用地图"""
    return list(MAPS.keys())


def get_edges(name: str) -> list[tuple[str, str]]:
    """从地图配置中提取边列表"""
    map_config = get_map(name)
    neighbors = map_config["neighbors"]
    edges = []
    seen = set()
    
    for u, nbs in neighbors.items():
        for v in nbs:
            edge = tuple(sorted([u, v]))
            if edge not in seen:
                edges.append((u, v))
                seen.add(edge)
    
    return edges


def get_color_domain(name: str) -> list[int]:
    """获取颜色域（整数索引）"""
    map_config = get_map(name)
    return list(range(len(map_config["colors"])))


def get_layout(name: str) -> Optional[dict[str, tuple[float, float]]]:
    """获取地图的预定义布局，如果没有则返回 None"""
    map_config = get_map(name)
    return map_config.get("layout")
