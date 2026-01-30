# Implementation Plan: map_coloring 仓库重构清理

## Overview

本计划将 map_coloring 仓库的重构遗留问题分解为可执行的编码任务。按照依赖顺序执行，确保每个步骤完成后仓库保持可运行状态。

## Tasks

- [x] 1. 创建 README.md 并修复包安装
  - [x] 1.1 创建 README.md 文件
    - 包含项目简介、安装说明、快速开始、目录结构
    - 说明 pgm-toolkit 可选依赖的安装方式
    - _Requirements: 1.1, 1.3_
  - [x] 1.2 验证 `pip install -e .` 可以成功执行
    - 确保 pyproject.toml 中引用的 README.md 存在
    - _Requirements: 1.2_

- [x] 2. 迁移 pgm-toolkit 到 external/ 目录
  - [x] 2.1 创建 external/ 目录并移动 pgm-toolkit
    - 移动 pgm-toolkit/ 到 external/pgm-toolkit/
    - 移除嵌套的 .git 目录（或转为 submodule）
    - _Requirements: 8.1, 8.4_
  - [x] 2.2 更新 pyproject.toml 文档说明
    - 添加可选依赖说明或安装指引
    - _Requirements: 8.2_

- [x] 3. 实现 core/inference 协议层
  - [x] 3.1 创建 core/inference/schemas.py
    - 实现 TraceStep dataclass
    - 实现 TraceSchema dataclass
    - 实现 ResultSchema dataclass
    - _Requirements: 3.1, 3.2_
  - [x] 3.2 创建 core/inference/adapters.py
    - 实现 InferenceAdapter 抽象基类
    - _Requirements: 3.3_
  - [x] 3.3 更新 core/inference/__init__.py 导出
    - 导出所有公共类
    - _Requirements: 3.1, 3.2, 3.3_

- [x] 4. Checkpoint - 确保包可安装
  - 执行 `pip install -e .` 验证安装
  - 验证 `from core.inference import TraceSchema, ResultSchema` 可导入
  - 如有问题请询问用户

- [x] 5. 消除 sys.path 注入并统一数据源
  - [x] 5.1 修改 scripts/run_comparison.py
    - 移除 sys.path.insert 语句
    - 改为从 tasks.experiment1_map_coloring.maps 导入 MAPS
    - _Requirements: 2.1, 6.3_
  - [x] 5.2 修改 scripts/quick_csp_demo.py
    - 移除 sys.path.insert 语句
    - 使用标准包导入
    - _Requirements: 2.2_
  - [x] 5.3 修改 scripts/quick_bp_demo.py
    - 移除 sys.path.insert 语句
    - 使用标准包导入
    - _Requirements: 2.3_
  - [x] 5.4 修改 tasks/experiment1_map_coloring/demo_csp_solver.py
    - 移除内嵌的 MAPS 字典定义
    - 改为从 maps.py 导入 MAPS
    - _Requirements: 6.1, 6.2_
  - [ ]* 5.5 编写属性测试：脚本无 sys.path 注入
    - **Property 1: 脚本无 sys.path 注入**
    - **Validates: Requirements 2.1, 2.2, 2.3**

- [x] 6. 修正 .gitignore 规则
  - [x] 6.1 更新 .gitignore 匹配实际 run_id 格式
    - 添加 `analysis/stats/**/20*/` 模式
    - 添加 `analysis/plots/**/20*/` 模式
    - 保留原有 `run_*/` 模式向后兼容
    - _Requirements: 5.1, 5.3, 5.4_

- [x] 7. 清理根目录产物
  - [x] 7.1 移动或删除根目录的 .png 文件
    - 将实验产物移至 analysis/plots/ 或删除
    - _Requirements: 9.1, 9.3_

- [x] 8. Checkpoint - 验证脚本可运行
  - 执行 `pip install -e .` 重新安装
  - 验证 scripts/ 下脚本可以正常导入和运行
  - 如有问题请询问用户

- [x] 9. 创建基础测试套件
  - [x] 9.1 创建 tests/test_csp_solver.py
    - 测试 CSP 求解器基本功能
    - 测试解的有效性（相邻不同色）
    - _Requirements: 4.1, 4.3_
  - [ ]* 9.2 编写属性测试：CSP 解有效性
    - **Property 3: CSP 解有效性（相邻不同色）**
    - **Validates: Requirements 4.3**
  - [x] 9.3 创建 tests/test_search_logger.py
    - 测试 SearchLogger 计数一致性
    - _Requirements: 4.4_
  - [ ]* 9.4 编写属性测试：SearchLogger 计数一致性
    - **Property 4: SearchLogger 计数一致性**
    - **Validates: Requirements 4.4**
  - [x] 9.5 创建 tests/test_bp_metrics.py
    - 测试 BP 指标计算（带 skip 机制）
    - 当 pgm-toolkit 不可用时跳过测试
    - _Requirements: 4.2, 4.5_

- [x] 10. 验证编码正确性
  - [x] 10.1 确保 maps.py 使用 UTF-8 编码保存
    - 验证中文字符正确显示
    - _Requirements: 7.1, 7.2_

- [x] 11. Final Checkpoint - 运行完整测试套件
  - 执行 `pytest tests/ -v` 验证所有测试通过
  - 验证 pgm-toolkit 缺失时 BP 测试正确跳过
  - 如有问题请询问用户

## Notes

- 任务按依赖顺序排列：先确保包可安装，再修改脚本，最后添加测试
- 标记 `*` 的子任务为可选属性测试任务
- 每个 Checkpoint 用于验证阶段性成果，确保增量可用
- pgm-toolkit 相关功能在依赖不可用时应优雅降级
