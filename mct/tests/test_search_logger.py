# -*- coding: utf-8 -*-
"""
SearchLogger 计数一致性测试

测试 SearchLogger 的计数一致性，验证：
1. step_count 等于 steps 列表长度
2. backtrack_count 等于 steps 中 'backtrack' 动作的数量

Requirements: 4.4
"""

import pytest
from map_coloring.agents.classical_search import (
    CSPSolver,
    CSPConfig,
    SearchLogger,
    SearchStepRecord,
)
from map_coloring.data.maps import MAPS, get_map


class TestSearchLoggerCountConsistency:
    """SearchLogger 计数一致性测试"""
    
    def test_step_count_equals_steps_length(self):
        """
        验证 step_count 等于 steps 列表长度
        
        **Validates: Requirements 4.4**
        """
        map_config = get_map("triangle")
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
        )
        
        solution = solver.solve()
        logger = solver.logger
        
        # 验证 step_count 等于 steps 列表长度
        assert logger.summary.total_steps == len(logger.steps), (
            f"step_count ({logger.summary.total_steps}) 应等于 "
            f"steps 列表长度 ({len(logger.steps)})"
        )
    
    def test_backtrack_count_equals_backtrack_events(self):
        """
        验证 backtrack_count 等于 backtrack_events 列表长度
        
        注意：total_backtracks 统计的是深度回退事件（通过 log_backtrack 记录），
        而不是 steps 中 action='backtrack' 的步骤数。
        action='backtrack' 表示某变量的所有颜色都已尝试完毕。
        
        **Validates: Requirements 4.4**
        """
        map_config = get_map("triangle")
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
        )
        
        solution = solver.solve()
        logger = solver.logger
        
        # 验证 backtrack_count 等于 backtrack_events 列表长度
        assert logger.summary.total_backtracks == len(logger.backtrack_events), (
            f"backtrack_count ({logger.summary.total_backtracks}) 应等于 "
            f"backtrack_events 列表长度 ({len(logger.backtrack_events)})"
        )
    
    def test_logger_consistency_australia_map(self):
        """
        测试 australia 地图的 logger 一致性
        
        **Validates: Requirements 4.4**
        """
        map_config = get_map("australia")
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
        )
        
        solution = solver.solve()
        logger = solver.logger
        
        # 验证 step_count 等于 steps 列表长度
        assert logger.summary.total_steps == len(logger.steps), (
            f"australia: step_count ({logger.summary.total_steps}) 应等于 "
            f"steps 列表长度 ({len(logger.steps)})"
        )
        
        # 验证 backtrack_count 等于 backtrack_events 列表长度
        assert logger.summary.total_backtracks == len(logger.backtrack_events), (
            f"australia: backtrack_count ({logger.summary.total_backtracks}) 应等于 "
            f"backtrack_events 列表长度 ({len(logger.backtrack_events)})"
        )
    
    def test_logger_consistency_k4_map(self):
        """
        测试 k4 地图的 logger 一致性
        
        K4 是完全图，需要4种颜色，搜索过程会有较多回溯。
        
        **Validates: Requirements 4.4**
        """
        map_config = get_map("k4")
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
        )
        
        solution = solver.solve()
        logger = solver.logger
        
        # 验证 step_count 等于 steps 列表长度
        assert logger.summary.total_steps == len(logger.steps), (
            f"k4: step_count ({logger.summary.total_steps}) 应等于 "
            f"steps 列表长度 ({len(logger.steps)})"
        )
        
        # 验证 backtrack_count 等于 backtrack_events 列表长度
        assert logger.summary.total_backtracks == len(logger.backtrack_events), (
            f"k4: backtrack_count ({logger.summary.total_backtracks}) 应等于 "
            f"backtrack_events 列表长度 ({len(logger.backtrack_events)})"
        )
    
    def test_logger_consistency_petersen_map(self):
        """
        测试 petersen 图的 logger 一致性
        
        彼得森图是经典的图论测试用例，搜索过程较复杂。
        
        **Validates: Requirements 4.4**
        """
        map_config = get_map("petersen")
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
        )
        
        solution = solver.solve()
        logger = solver.logger
        
        # 验证 step_count 等于 steps 列表长度
        assert logger.summary.total_steps == len(logger.steps), (
            f"petersen: step_count ({logger.summary.total_steps}) 应等于 "
            f"steps 列表长度 ({len(logger.steps)})"
        )
        
        # 验证 backtrack_count 等于 backtrack_events 列表长度
        assert logger.summary.total_backtracks == len(logger.backtrack_events), (
            f"petersen: backtrack_count ({logger.summary.total_backtracks}) 应等于 "
            f"backtrack_events 列表长度 ({len(logger.backtrack_events)})"
        )
    
    def test_logger_consistency_no_solution_case(self):
        """
        测试无解情况下的 logger 一致性
        
        K3（三角形）使用2色无解，会产生大量回溯。
        
        **Validates: Requirements 4.4**
        """
        map_config = get_map("triangle")
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=["红", "绿"],  # 只有2种颜色，无解
        )
        
        solution = solver.solve()
        logger = solver.logger
        
        # 应该无解
        assert solution is None, "K3 使用2色应该无解"
        
        # 验证 step_count 等于 steps 列表长度
        assert logger.summary.total_steps == len(logger.steps), (
            f"无解情况: step_count ({logger.summary.total_steps}) 应等于 "
            f"steps 列表长度 ({len(logger.steps)})"
        )
        
        # 验证 backtrack_count 等于 backtrack_events 列表长度
        assert logger.summary.total_backtracks == len(logger.backtrack_events), (
            f"无解情况: backtrack_count ({logger.summary.total_backtracks}) 应等于 "
            f"backtrack_events 列表长度 ({len(logger.backtrack_events)})"
        )


class TestSearchLoggerAllMaps:
    """测试所有预定义地图的 logger 一致性"""
    
    @pytest.mark.parametrize("map_name", list(MAPS.keys()))
    def test_logger_consistency_all_maps(self, map_name: str):
        """
        测试所有预定义地图的 logger 一致性
        
        遍历 MAPS 中的所有地图，验证 SearchLogger 计数一致性。
        
        **Validates: Requirements 4.4**
        """
        map_config = get_map(map_name)
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
        )
        
        solution = solver.solve()
        logger = solver.logger
        
        # 验证 step_count 等于 steps 列表长度
        assert logger.summary.total_steps == len(logger.steps), (
            f"地图 {map_name}: step_count ({logger.summary.total_steps}) 应等于 "
            f"steps 列表长度 ({len(logger.steps)})"
        )
        
        # 验证 backtrack_count 等于 backtrack_events 列表长度
        assert logger.summary.total_backtracks == len(logger.backtrack_events), (
            f"地图 {map_name}: backtrack_count ({logger.summary.total_backtracks}) 应等于 "
            f"backtrack_events 列表长度 ({len(logger.backtrack_events)})"
        )


class TestSearchLoggerDifferentStrategies:
    """测试不同求解策略下的 logger 一致性"""
    
    @pytest.mark.parametrize("var_strategy", ["static", "mrv", "degree", "mrv+degree"])
    def test_logger_consistency_var_strategies(self, var_strategy: str):
        """
        测试不同变量排序策略下的 logger 一致性
        
        **Validates: Requirements 4.4**
        """
        map_config = get_map("australia")
        config = CSPConfig(var_ordering_strategy=var_strategy)
        
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
            config=config,
        )
        
        solution = solver.solve()
        logger = solver.logger
        
        # 验证 step_count 等于 steps 列表长度
        assert logger.summary.total_steps == len(logger.steps), (
            f"策略 {var_strategy}: step_count ({logger.summary.total_steps}) 应等于 "
            f"steps 列表长度 ({len(logger.steps)})"
        )
        
        # 验证 backtrack_count 等于 backtrack_events 列表长度
        assert logger.summary.total_backtracks == len(logger.backtrack_events), (
            f"策略 {var_strategy}: backtrack_count ({logger.summary.total_backtracks}) 应等于 "
            f"backtrack_events 列表长度 ({len(logger.backtrack_events)})"
        )
    
    def test_logger_consistency_forward_checking(self):
        """
        测试启用前向检查时的 logger 一致性
        
        **Validates: Requirements 4.4**
        """
        map_config = get_map("australia")
        config = CSPConfig(use_forward_checking=True)
        
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
            config=config,
        )
        
        solution = solver.solve()
        logger = solver.logger
        
        # 验证 step_count 等于 steps 列表长度
        assert logger.summary.total_steps == len(logger.steps), (
            f"前向检查: step_count ({logger.summary.total_steps}) 应等于 "
            f"steps 列表长度 ({len(logger.steps)})"
        )
        
        # 验证 backtrack_count 等于 backtrack_events 列表长度
        assert logger.summary.total_backtracks == len(logger.backtrack_events), (
            f"前向检查: backtrack_count ({logger.summary.total_backtracks}) 应等于 "
            f"backtrack_events 列表长度 ({len(logger.backtrack_events)})"
        )


class TestSearchLoggerAdditionalInvariants:
    """测试 SearchLogger 的其他不变量"""
    
    def test_assign_count_consistency(self):
        """
        验证 total_assigns 等于 steps 中 'assign' 动作的数量
        
        **Validates: Requirements 4.4**
        """
        map_config = get_map("australia")
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
        )
        
        solution = solver.solve()
        logger = solver.logger
        
        # 统计 steps 中 'assign' 动作的数量
        assign_actions_count = sum(
            1 for step in logger.steps if step.action == "assign"
        )
        
        # 验证 total_assigns 等于 assign 动作数量
        assert logger.summary.total_assigns == assign_actions_count, (
            f"total_assigns ({logger.summary.total_assigns}) 应等于 "
            f"steps 中 'assign' 动作数量 ({assign_actions_count})"
        )
    
    def test_reject_count_consistency(self):
        """
        验证 total_rejects 等于 steps 中 'reject' 动作的数量
        
        **Validates: Requirements 4.4**
        """
        map_config = get_map("australia")
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
        )
        
        solution = solver.solve()
        logger = solver.logger
        
        # 统计 steps 中 'reject' 动作的数量
        reject_actions_count = sum(
            1 for step in logger.steps if step.action == "reject"
        )
        
        # 验证 total_rejects 等于 reject 动作数量
        assert logger.summary.total_rejects == reject_actions_count, (
            f"total_rejects ({logger.summary.total_rejects}) 应等于 "
            f"steps 中 'reject' 动作数量 ({reject_actions_count})"
        )
    
    def test_step_ids_are_sequential(self):
        """
        验证 step_id 是连续递增的
        
        **Validates: Requirements 4.4**
        """
        map_config = get_map("australia")
        solver = CSPSolver(
            regions=map_config["regions"],
            neighbors=map_config["neighbors"],
            colors=map_config["colors"],
        )
        
        solution = solver.solve()
        logger = solver.logger
        
        # 验证 step_id 是从 1 开始连续递增的
        for i, step in enumerate(logger.steps):
            expected_id = i + 1
            assert step.step_id == expected_id, (
                f"step_id 应为 {expected_id}，实际为 {step.step_id}"
            )
