# -*- coding: utf-8 -*-
"""
CSP 求解器单元测试

测试 CSP 求解器的基本功能和解的有效性。

Requirements: 4.1, 4.3
"""

import pytest
from map_coloring.agents.classical_search import CSPSolver, CSPConfig
from map_coloring.data.maps import MAPS, get_map


def is_valid_solution(
    solution: dict[str, str],
    neighbors: dict[str, list[str]],
    regions: list[str],
) -> bool:
    """
    验证解的有效性：相邻区域不同色
    
    Args:
        solution: 求解结果（区域 -> 颜色映射）
        neighbors: 邻接关系字典
        regions: 所有区域列表
    
    Returns:
        True 如果解有效，False 否则
    """
    # 检查所有区域都被赋值
    if set(solution.keys()) != set(regions):
        return False
    
    # 检查相邻区域颜色不同
    for region, region_neighbors in neighbors.items():
        for neighbor in region_neighbors:
            if solution.get(region) == solution.get(neighbor):
                return False
    
    return True


class TestCSPSolverBasic:
    """CSP 求解器基本功能测试"""
    
    def test_solver_initialization(self):
        """测试求解器初始化"""
        map_config = get_map("triangle")
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
        )
        
        assert solver.regions == map_config["regions"]
        assert solver.neighbors == map_config["neighbors"]
        assert solver.colors == map_config["colors"]
        assert solver.solution is None
    
    def test_solver_with_custom_config(self):
        """测试使用自定义配置初始化求解器"""
        map_config = get_map("triangle")
        config = CSPConfig(
            var_ordering_strategy="mrv",
            use_forward_checking=True,
        )
        
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
            config=config,
        )
        
        assert solver.config.var_ordering_strategy == "mrv"
        assert solver.config.use_forward_checking is True
    
    def test_invalid_config_raises_error(self):
        """测试无效配置抛出错误"""
        map_config = get_map("triangle")
        config = CSPConfig(max_steps=-1)  # 无效配置
        
        with pytest.raises(ValueError, match="配置错误"):
            CSPSolver(
                regions=map_config["regions"],
                neighbors=map_config["neighbors"],
                colors=map_config["colors"],
                config=config,
            )


class TestSolutionValidity:
    """验证解的有效性：相邻区域不同色"""
    
    def test_triangle_map_solution_validity(self):
        """
        测试 triangle 地图（3色足够）
        
        triangle 是一个简单的三角形图，3个节点全连接，
        需要3种颜色才能着色。
        """
        map_config = get_map("triangle")
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
        )
        
        solution = solver.solve()
        
        # 应该找到解
        assert solution is not None, "triangle 地图应该有解"
        
        # 验证解的有效性
        assert is_valid_solution(
            solution,
            map_config["neighbors"],
            map_config["regions"],
        ), f"解无效：存在相邻区域同色。解: {solution}"
    
    def test_australia_map_solution_validity(self):
        """
        测试 australia 地图
        
        澳大利亚地图是经典的 CSP 问题，7个区域，
        使用3种颜色可以着色。
        """
        map_config = get_map("australia")
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
        )
        
        solution = solver.solve()
        
        # 应该找到解
        assert solution is not None, "australia 地图应该有解"
        
        # 验证解的有效性
        assert is_valid_solution(
            solution,
            map_config["neighbors"],
            map_config["regions"],
        ), f"解无效：存在相邻区域同色。解: {solution}"
    
    def test_k4_map_solution_validity(self):
        """
        测试 k4 地图（需要4色）
        
        K4 是完全图，4个节点全连接，
        需要4种颜色才能着色（色数 = 4）。
        """
        map_config = get_map("k4")
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
        )
        
        solution = solver.solve()
        
        # 应该找到解
        assert solution is not None, "k4 地图应该有解（使用4色）"
        
        # 验证解的有效性
        assert is_valid_solution(
            solution,
            map_config["neighbors"],
            map_config["regions"],
        ), f"解无效：存在相邻区域同色。解: {solution}"
    
    def test_square_map_solution_validity(self):
        """
        测试 square 地图（2色足够）
        
        square 是一个4节点的环形图，
        只需要2种颜色即可着色。
        """
        map_config = get_map("square")
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
        )
        
        solution = solver.solve()
        
        # 应该找到解
        assert solution is not None, "square 地图应该有解"
        
        # 验证解的有效性
        assert is_valid_solution(
            solution,
            map_config["neighbors"],
            map_config["regions"],
        ), f"解无效：存在相邻区域同色。解: {solution}"
    
    def test_petersen_map_solution_validity(self):
        """
        测试 petersen 图（经典难图）
        
        彼得森图是一个经典的图论测试用例，
        色数为3。
        """
        map_config = get_map("petersen")
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
        )
        
        solution = solver.solve()
        
        # 应该找到解
        assert solution is not None, "petersen 图应该有解"
        
        # 验证解的有效性
        assert is_valid_solution(
            solution,
            map_config["neighbors"],
            map_config["regions"],
        ), f"解无效：存在相邻区域同色。解: {solution}"


class TestNoSolutionCases:
    """测试无解情况"""
    
    def test_k3_with_2_colors_no_solution(self):
        """
        测试 K3（三角形）使用2色无解
        
        完全图 K3 需要3种颜色，使用2种颜色应该无解。
        """
        # 使用 triangle 地图但只给2种颜色
        map_config = get_map("triangle")
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=["红", "绿"],  # 只有2种颜色
        )
        
        solution = solver.solve()
        
        # 应该无解
        assert solution is None, "K3 使用2色应该无解"
    
    def test_k4_with_3_colors_no_solution(self):
        """
        测试 K4 使用3色无解
        
        完全图 K4 需要4种颜色，使用3种颜色应该无解。
        """
        map_config = get_map("k4")
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=["红", "绿", "蓝"],  # 只有3种颜色
        )
        
        solution = solver.solve()
        
        # 应该无解
        assert solution is None, "K4 使用3色应该无解"


class TestDifferentStrategies:
    """测试不同求解策略"""
    
    @pytest.mark.parametrize("var_strategy", ["static", "mrv", "degree", "mrv+degree"])
    def test_var_ordering_strategies(self, var_strategy: str):
        """测试不同变量排序策略都能找到有效解"""
        map_config = get_map("australia")
        config = CSPConfig(var_ordering_strategy=var_strategy)
        
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
            config=config,
        )
        
        solution = solver.solve()
        
        assert solution is not None, f"策略 {var_strategy} 应该找到解"
        assert is_valid_solution(
            solution,
            map_config["neighbors"],
            map_config["regions"],
        ), f"策略 {var_strategy} 的解无效"
    
    @pytest.mark.parametrize("value_strategy", ["static", "least_constraining"])
    def test_value_ordering_strategies(self, value_strategy: str):
        """测试不同值排序策略都能找到有效解"""
        map_config = get_map("australia")
        config = CSPConfig(value_ordering_strategy=value_strategy)
        
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
            config=config,
        )
        
        solution = solver.solve()
        
        assert solution is not None, f"策略 {value_strategy} 应该找到解"
        assert is_valid_solution(
            solution,
            map_config["neighbors"],
            map_config["regions"],
        ), f"策略 {value_strategy} 的解无效"
    
    def test_forward_checking_enabled(self):
        """测试启用前向检查能找到有效解"""
        map_config = get_map("australia")
        config = CSPConfig(use_forward_checking=True)
        
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
            config=config,
        )
        
        solution = solver.solve()
        
        assert solution is not None, "启用前向检查应该找到解"
        assert is_valid_solution(
            solution,
            map_config["neighbors"],
            map_config["regions"],
        ), "启用前向检查的解无效"
    
    def test_combined_strategies(self):
        """测试组合策略（MRV + 前向检查 + 最少约束值）"""
        map_config = get_map("petersen")
        config = CSPConfig(
            var_ordering_strategy="mrv",
            value_ordering_strategy="least_constraining",
            use_forward_checking=True,
        )
        
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
            config=config,
        )
        
        solution = solver.solve()
        
        assert solution is not None, "组合策略应该找到解"
        assert is_valid_solution(
            solution,
            map_config["neighbors"],
            map_config["regions"],
        ), "组合策略的解无效"


class TestFindAllSolutions:
    """测试查找所有解功能"""
    
    def test_find_all_solutions_triangle(self):
        """测试查找 triangle 地图的所有解"""
        map_config = get_map("triangle")
        config = CSPConfig(find_all_solutions=True)
        
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
            config=config,
        )
        
        solution = solver.solve()
        
        # 应该找到解
        assert solution is not None
        
        # 应该找到多个解（3! = 6 种颜色排列）
        assert len(solver.all_solutions) == 6, \
            f"triangle 应该有6个解，实际找到 {len(solver.all_solutions)} 个"
        
        # 验证所有解都有效
        for sol in solver.all_solutions:
            assert is_valid_solution(
                sol,
                map_config["neighbors"],
                map_config["regions"],
            ), f"解无效: {sol}"


class TestAllMaps:
    """测试所有预定义地图"""
    
    @pytest.mark.parametrize("map_name", list(MAPS.keys()))
    def test_all_maps_have_valid_solutions(self, map_name: str):
        """
        测试所有预定义地图都能找到有效解
        
        遍历 MAPS 中的所有地图，验证求解器能找到有效解。
        """
        map_config = get_map(map_name)
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
        )
        
        solution = solver.solve()
        
        assert solution is not None, f"地图 {map_name} 应该有解"
        assert is_valid_solution(
            solution,
            map_config["neighbors"],
            map_config["regions"],
        ), f"地图 {map_name} 的解无效: {solution}"
