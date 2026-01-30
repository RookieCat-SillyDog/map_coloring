# -*- coding: utf-8 -*-
"""
BP 指标计算测试

测试 BP（置信传播）指标计算的基本功能，包括：
- compute_expected_violations: 计算期望冲突数
- compute_map_conflicts: 计算 MAP 冲突数
- MetricsCalculator: 度量计算器类

当 pgm-toolkit 不可用时，测试会优雅跳过并提供清晰的跳过原因消息。

Requirements: 4.2, 4.5
"""

import pytest
import numpy as np

# 尝试导入 BP 相关模块，如果 pgm-toolkit 不可用则标记为不可用
try:
    from map_coloring.agents.bayesian.research_bp import (
        compute_expected_violations,
        compute_map_conflicts,
        MetricsCalculator,
    )
    from pgm_toolkit.core.graph import Factor
    PGM_AVAILABLE = True
    PGM_IMPORT_ERROR = None
except ImportError as e:
    PGM_AVAILABLE = False
    PGM_IMPORT_ERROR = str(e)
    # 定义占位符以避免 NameError
    compute_expected_violations = None
    compute_map_conflicts = None
    MetricsCalculator = None
    Factor = None


# 跳过原因消息
SKIP_REASON = (
    "pgm-toolkit not installed. "
    "Install with: pip install -e external/pgm-toolkit"
    + (f" (Import error: {PGM_IMPORT_ERROR})" if PGM_IMPORT_ERROR else "")
)


@pytest.mark.skipif(not PGM_AVAILABLE, reason=SKIP_REASON)
class TestComputeExpectedViolations:
    """测试 compute_expected_violations 函数"""
    
    def test_empty_graph(self):
        """
        测试空图（无边）的期望冲突数
        
        空图没有边，期望冲突数应为 0。
        
        **Validates: Requirements 4.2**
        """
        edges = []
        beliefs = {"A": np.array([0.5, 0.5]), "B": np.array([0.5, 0.5])}
        
        result = compute_expected_violations(edges, beliefs)
        
        assert result == 0.0, f"空图的期望冲突数应为 0，实际为 {result}"
    
    def test_single_edge_uniform_beliefs(self):
        """
        测试单边图的均匀信念期望冲突数
        
        两个节点各有2种颜色，均匀分布时：
        E_viol = sum_c b_A(c) * b_B(c) = 0.5 * 0.5 + 0.5 * 0.5 = 0.5
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B")]
        beliefs = {
            "A": np.array([0.5, 0.5]),
            "B": np.array([0.5, 0.5]),
        }
        
        result = compute_expected_violations(edges, beliefs)
        
        assert abs(result - 0.5) < 1e-9, f"期望冲突数应为 0.5，实际为 {result}"
    
    def test_single_edge_deterministic_same_color(self):
        """
        测试单边图的确定性信念（相同颜色）
        
        两个节点都确定选择颜色 0：
        E_viol = 1.0 * 1.0 = 1.0
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B")]
        beliefs = {
            "A": np.array([1.0, 0.0]),
            "B": np.array([1.0, 0.0]),
        }
        
        result = compute_expected_violations(edges, beliefs)
        
        assert abs(result - 1.0) < 1e-9, f"期望冲突数应为 1.0，实际为 {result}"
    
    def test_single_edge_deterministic_different_colors(self):
        """
        测试单边图的确定性信念（不同颜色）
        
        节点 A 选择颜色 0，节点 B 选择颜色 1：
        E_viol = 1.0 * 0.0 + 0.0 * 1.0 = 0.0
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B")]
        beliefs = {
            "A": np.array([1.0, 0.0]),
            "B": np.array([0.0, 1.0]),
        }
        
        result = compute_expected_violations(edges, beliefs)
        
        assert abs(result - 0.0) < 1e-9, f"期望冲突数应为 0.0，实际为 {result}"
    
    def test_triangle_graph_uniform_beliefs(self):
        """
        测试三角形图（K3）的均匀信念期望冲突数
        
        三个节点各有3种颜色，均匀分布时：
        每条边的期望冲突 = sum_c (1/3) * (1/3) = 3 * (1/9) = 1/3
        总期望冲突 = 3 * (1/3) = 1.0
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B"), ("B", "C"), ("A", "C")]
        beliefs = {
            "A": np.array([1/3, 1/3, 1/3]),
            "B": np.array([1/3, 1/3, 1/3]),
            "C": np.array([1/3, 1/3, 1/3]),
        }
        
        result = compute_expected_violations(edges, beliefs)
        
        assert abs(result - 1.0) < 1e-9, f"期望冲突数应为 1.0，实际为 {result}"
    
    def test_missing_belief_node_skipped(self):
        """
        测试缺失信念的节点被跳过
        
        如果边的某个端点没有信念信息，该边应被跳过。
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B"), ("B", "C")]
        beliefs = {
            "A": np.array([0.5, 0.5]),
            "B": np.array([0.5, 0.5]),
            # C 没有信念
        }
        
        result = compute_expected_violations(edges, beliefs)
        
        # 只有 (A, B) 边被计算，(B, C) 被跳过
        expected = 0.5  # A-B 边的期望冲突
        assert abs(result - expected) < 1e-9, f"期望冲突数应为 {expected}，实际为 {result}"
    
    def test_unnormalized_beliefs_are_normalized(self):
        """
        测试未归一化的信念会被自动归一化
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B")]
        beliefs = {
            "A": np.array([2.0, 2.0]),  # 未归一化，和为 4
            "B": np.array([1.0, 1.0]),  # 未归一化，和为 2
        }
        
        result = compute_expected_violations(edges, beliefs)
        
        # 归一化后等价于均匀分布
        expected = 0.5
        assert abs(result - expected) < 1e-9, f"期望冲突数应为 {expected}，实际为 {result}"


@pytest.mark.skipif(not PGM_AVAILABLE, reason=SKIP_REASON)
class TestComputeMapConflicts:
    """测试 compute_map_conflicts 函数"""
    
    def test_empty_graph(self):
        """
        测试空图（无边）的 MAP 冲突数
        
        空图没有边，MAP 冲突数应为 0。
        
        **Validates: Requirements 4.2**
        """
        edges = []
        beliefs = {"A": np.array([0.5, 0.5]), "B": np.array([0.5, 0.5])}
        
        result = compute_map_conflicts(edges, beliefs)
        
        assert result == 0, f"空图的 MAP 冲突数应为 0，实际为 {result}"
    
    def test_single_edge_same_map_assignment(self):
        """
        测试单边图的相同 MAP 赋值
        
        两个节点的 MAP 估计都是颜色 0，产生 1 个冲突。
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B")]
        beliefs = {
            "A": np.array([0.8, 0.2]),  # MAP: 颜色 0
            "B": np.array([0.7, 0.3]),  # MAP: 颜色 0
        }
        
        result = compute_map_conflicts(edges, beliefs)
        
        assert result == 1, f"MAP 冲突数应为 1，实际为 {result}"
    
    def test_single_edge_different_map_assignment(self):
        """
        测试单边图的不同 MAP 赋值
        
        节点 A 的 MAP 是颜色 0，节点 B 的 MAP 是颜色 1，无冲突。
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B")]
        beliefs = {
            "A": np.array([0.8, 0.2]),  # MAP: 颜色 0
            "B": np.array([0.3, 0.7]),  # MAP: 颜色 1
        }
        
        result = compute_map_conflicts(edges, beliefs)
        
        assert result == 0, f"MAP 冲突数应为 0，实际为 {result}"
    
    def test_triangle_graph_all_same_color(self):
        """
        测试三角形图（K3）所有节点选择相同颜色
        
        三个节点的 MAP 估计都是颜色 0，产生 3 个冲突。
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B"), ("B", "C"), ("A", "C")]
        beliefs = {
            "A": np.array([0.9, 0.05, 0.05]),  # MAP: 颜色 0
            "B": np.array([0.8, 0.1, 0.1]),    # MAP: 颜色 0
            "C": np.array([0.7, 0.15, 0.15]),  # MAP: 颜色 0
        }
        
        result = compute_map_conflicts(edges, beliefs)
        
        assert result == 3, f"MAP 冲突数应为 3，实际为 {result}"
    
    def test_triangle_graph_valid_coloring(self):
        """
        测试三角形图（K3）的有效着色
        
        三个节点分别选择不同颜色，无冲突。
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B"), ("B", "C"), ("A", "C")]
        beliefs = {
            "A": np.array([0.9, 0.05, 0.05]),  # MAP: 颜色 0
            "B": np.array([0.05, 0.9, 0.05]),  # MAP: 颜色 1
            "C": np.array([0.05, 0.05, 0.9]),  # MAP: 颜色 2
        }
        
        result = compute_map_conflicts(edges, beliefs)
        
        assert result == 0, f"MAP 冲突数应为 0，实际为 {result}"
    
    def test_with_evidence(self):
        """
        测试带证据的 MAP 冲突计算
        
        证据会覆盖信念的 MAP 估计。
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B")]
        beliefs = {
            "A": np.array([0.8, 0.2]),  # MAP: 颜色 0
            "B": np.array([0.7, 0.3]),  # MAP: 颜色 0
        }
        evidence = {"A": 1}  # 强制 A 为颜色 1
        
        result = compute_map_conflicts(edges, beliefs, evidence=evidence)
        
        # A=1, B=0 (MAP)，不同颜色，无冲突
        assert result == 0, f"MAP 冲突数应为 0，实际为 {result}"
    
    def test_with_domains(self):
        """
        测试带域信息的 MAP 冲突计算
        
        域信息用于将索引映射到实际颜色值。
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B")]
        beliefs = {
            "A": np.array([0.8, 0.2]),  # MAP: 索引 0 -> "红"
            "B": np.array([0.7, 0.3]),  # MAP: 索引 0 -> "红"
        }
        domains = {
            "A": ["红", "绿"],
            "B": ["红", "绿"],
        }
        
        result = compute_map_conflicts(edges, beliefs, domains=domains)
        
        # 两个节点都选择 "红"，产生 1 个冲突
        assert result == 1, f"MAP 冲突数应为 1，实际为 {result}"
    
    def test_missing_belief_node(self):
        """
        测试缺失信念的节点
        
        如果边的某个端点没有信念信息，该边不计入冲突。
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B"), ("B", "C")]
        beliefs = {
            "A": np.array([0.8, 0.2]),  # MAP: 颜色 0
            "B": np.array([0.7, 0.3]),  # MAP: 颜色 0
            # C 没有信念
        }
        
        result = compute_map_conflicts(edges, beliefs)
        
        # 只有 (A, B) 边被计算，A=0, B=0，产生 1 个冲突
        # (B, C) 边因 C 没有赋值而不计入
        assert result == 1, f"MAP 冲突数应为 1，实际为 {result}"


@pytest.mark.skipif(not PGM_AVAILABLE, reason=SKIP_REASON)
class TestMetricsCalculator:
    """测试 MetricsCalculator 类"""
    
    def test_initialization(self):
        """
        测试 MetricsCalculator 初始化
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B"), ("B", "C")]
        calculator = MetricsCalculator(edges)
        
        assert calculator.edges == [("A", "B"), ("B", "C")]
    
    def test_compute_returns_both_metrics(self):
        """
        测试 compute 方法返回两种度量
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B")]
        calculator = MetricsCalculator(edges)
        beliefs = {
            "A": np.array([0.5, 0.5]),
            "B": np.array([0.5, 0.5]),
        }
        
        result = calculator.compute(beliefs)
        
        assert "E_viol" in result, "结果应包含 E_viol"
        assert "MAP_conflicts" in result, "结果应包含 MAP_conflicts"
    
    def test_compute_e_viol_value(self):
        """
        测试 compute 方法的 E_viol 值
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B")]
        calculator = MetricsCalculator(edges)
        beliefs = {
            "A": np.array([0.5, 0.5]),
            "B": np.array([0.5, 0.5]),
        }
        
        result = calculator.compute(beliefs)
        
        assert abs(result["E_viol"] - 0.5) < 1e-9, \
            f"E_viol 应为 0.5，实际为 {result['E_viol']}"
    
    def test_compute_map_conflicts_value(self):
        """
        测试 compute 方法的 MAP_conflicts 值
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B")]
        calculator = MetricsCalculator(edges)
        beliefs = {
            "A": np.array([0.8, 0.2]),  # MAP: 颜色 0
            "B": np.array([0.7, 0.3]),  # MAP: 颜色 0
        }
        
        result = calculator.compute(beliefs)
        
        assert result["MAP_conflicts"] == 1, \
            f"MAP_conflicts 应为 1，实际为 {result['MAP_conflicts']}"
    
    def test_compute_with_evidence_and_domains(self):
        """
        测试 compute 方法带证据和域信息
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B")]
        calculator = MetricsCalculator(edges)
        beliefs = {
            "A": np.array([0.8, 0.2]),
            "B": np.array([0.7, 0.3]),
        }
        evidence = {"A": 1}  # 强制 A 为颜色 1
        domains = {"A": ["红", "绿"], "B": ["红", "绿"]}
        
        result = calculator.compute(beliefs, evidence=evidence, domains=domains)
        
        # A=1 (绿), B=0 (红)，不同颜色，无冲突
        assert result["MAP_conflicts"] == 0, \
            f"MAP_conflicts 应为 0，实际为 {result['MAP_conflicts']}"


@pytest.mark.skipif(not PGM_AVAILABLE, reason=SKIP_REASON)
class TestWithFactorBeliefs:
    """测试使用 Factor 对象作为信念的情况"""
    
    def test_expected_violations_with_factor(self):
        """
        测试使用 Factor 对象计算期望冲突数
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B")]
        
        # 创建 Factor 对象作为信念
        # Factor 签名: Factor(scope, table, var_domains)
        factor_a = Factor(["A"], np.array([0.5, 0.5]), {"A": [0, 1]})
        factor_b = Factor(["B"], np.array([0.5, 0.5]), {"B": [0, 1]})
        
        beliefs = {"A": factor_a, "B": factor_b}
        
        result = compute_expected_violations(edges, beliefs)
        
        assert abs(result - 0.5) < 1e-9, f"期望冲突数应为 0.5，实际为 {result}"
    
    def test_map_conflicts_with_factor(self):
        """
        测试使用 Factor 对象计算 MAP 冲突数
        
        **Validates: Requirements 4.2**
        """
        edges = [("A", "B")]
        
        # 创建 Factor 对象作为信念
        # Factor 签名: Factor(scope, table, var_domains)
        factor_a = Factor(["A"], np.array([0.8, 0.2]), {"A": [0, 1]})
        factor_b = Factor(["B"], np.array([0.7, 0.3]), {"B": [0, 1]})
        
        beliefs = {"A": factor_a, "B": factor_b}
        
        result = compute_map_conflicts(edges, beliefs)
        
        # 两个节点的 MAP 都是 0，产生 1 个冲突
        assert result == 1, f"MAP 冲突数应为 1，实际为 {result}"


class TestSkipMechanism:
    """测试跳过机制"""
    
    def test_pgm_availability_flag_exists(self):
        """
        验证 PGM_AVAILABLE 标志存在
        
        **Validates: Requirements 4.5**
        """
        assert isinstance(PGM_AVAILABLE, bool), "PGM_AVAILABLE 应为布尔值"
    
    def test_skip_reason_is_informative(self):
        """
        验证跳过原因消息是有意义的
        
        **Validates: Requirements 4.5**
        """
        assert "pgm-toolkit" in SKIP_REASON.lower(), \
            "跳过原因应提及 pgm-toolkit"
        assert "install" in SKIP_REASON.lower(), \
            "跳过原因应包含安装指引"
