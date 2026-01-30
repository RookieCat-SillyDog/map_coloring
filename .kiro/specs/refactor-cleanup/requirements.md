# Requirements Document

## Introduction

本文档定义了 map_coloring 仓库重构遗留问题的修复需求。目标是消除当前仓库中违反 AGENTS.md 规范的问题，确保仓库达到"可安装、可运行、可测试"的工程标准。

## Glossary

- **Map_Coloring_Package**: 本仓库的可安装 Python 包，包含 core、agents、tasks、analysis、fitting、utils 模块
- **Sys_Path_Injection**: 通过 `sys.path.insert/append` 手动添加路径的做法，违反 AGENTS.md 硬约束
- **Trace_Schema**: 推理过程的追踪协议，记录算法执行的中间状态
- **Result_Schema**: 推理结果的标准化数据结构
- **Adapter**: 第三方库（如 pgm-toolkit）与本项目的适配层
- **Run_ID**: 实验运行的唯一标识符，格式为 `YYYYMMDD-HHMMSS_<note>`
- **PGM_Toolkit**: 概率图模型工具包，作为外部依赖引入

## Requirements

### Requirement 1: README.md 文件创建

**User Story:** As a developer, I want a README.md file to exist, so that the package can be installed and published without errors.

#### Acceptance Criteria

1. THE Map_Coloring_Package SHALL include a README.md file at the repository root
2. WHEN `pip install -e .` is executed, THE Map_Coloring_Package SHALL install successfully without file-not-found errors
3. THE README.md SHALL contain project description, installation instructions, and basic usage examples

### Requirement 2: 消除 sys.path 注入

**User Story:** As a developer, I want all scripts to work without sys.path manipulation, so that the codebase follows AGENTS.md constraints and is properly installable.

#### Acceptance Criteria

1. THE scripts/run_comparison.py SHALL NOT contain any `sys.path.insert` or `sys.path.append` statements
2. THE scripts/quick_csp_demo.py SHALL NOT contain any `sys.path.insert` or `sys.path.append` statements
3. THE scripts/quick_bp_demo.py SHALL NOT contain any `sys.path.insert` or `sys.path.append` statements
4. WHEN the package is installed via `pip install -e .`, THE scripts SHALL import modules using standard Python imports
5. WHEN any script in scripts/ is executed, THE script SHALL run successfully using installed package imports

### Requirement 3: core/inference 协议实现

**User Story:** As a developer, I want core/inference to contain adapter and trace/result schemas, so that the five-layer architecture is properly implemented.

#### Acceptance Criteria

1. THE core/inference module SHALL define a TraceSchema dataclass for recording inference steps
2. THE core/inference module SHALL define a ResultSchema dataclass for standardizing inference results
3. THE core/inference module SHALL define an InferenceAdapter abstract base class for third-party library integration
4. WHEN an algorithm produces results, THE results SHALL conform to the ResultSchema structure
5. WHEN an algorithm executes, THE trace SHALL conform to the TraceSchema structure

### Requirement 4: 基础测试套件

**User Story:** As a developer, I want basic tests to exist, so that refactoring does not break core functionality.

#### Acceptance Criteria

1. THE tests/ directory SHALL contain test files for CSP solver functionality
2. THE tests/ directory SHALL contain test files for BP metrics computation
3. WHEN CSP solver produces a solution, THE test SHALL verify that no adjacent regions share the same color
4. WHEN SearchLogger records steps, THE test SHALL verify that step counts are consistent
5. IF pgm_toolkit is not available, THEN THE BP tests SHALL skip gracefully with appropriate message
6. WHEN `pytest tests/` is executed, THE test suite SHALL run without import errors

### Requirement 5: .gitignore 规则修正

**User Story:** As a developer, I want .gitignore to correctly match the run_id format, so that experiment outputs are not accidentally committed.

#### Acceptance Criteria

1. THE .gitignore SHALL contain patterns that match the actual run_id format `YYYYMMDD-HHMMSS*`
2. WHEN a new experiment run creates output files, THE output files SHALL be ignored by git
3. THE .gitignore SHALL ignore `analysis/stats/**/20*` pattern for timestamp-based directories
4. THE .gitignore SHALL ignore `analysis/plots/**/20*` pattern for timestamp-based directories

### Requirement 6: 消除任务数据重复

**User Story:** As a developer, I want map data to be defined in a single location, so that the codebase follows DRY principle and Phase 2 decoupling is complete.

#### Acceptance Criteria

1. THE tasks/experiment1_map_coloring/demo_csp_solver.py SHALL import MAPS from tasks/experiment1_map_coloring/maps.py
2. THE tasks/experiment1_map_coloring/demo_csp_solver.py SHALL NOT define its own MAPS dictionary
3. THE scripts/run_comparison.py SHALL import MAPS from tasks/experiment1_map_coloring/maps.py
4. WHEN MAPS is modified in maps.py, THE change SHALL be reflected in all consumers without additional modifications

### Requirement 7: 编码问题修复

**User Story:** As a developer, I want all source files to use UTF-8 encoding correctly, so that Chinese characters display properly.

#### Acceptance Criteria

1. THE tasks/experiment1_map_coloring/maps.py SHALL be saved with UTF-8 encoding
2. WHEN maps.py is read by Python, THE Chinese characters SHALL display correctly without mojibake
3. THE pyproject.toml SHALL specify UTF-8 encoding for the project if applicable

### Requirement 8: pgm-toolkit 依赖规范化

**User Story:** As a developer, I want pgm-toolkit to be properly configured as an external dependency, so that it follows AGENTS.md方案A guidelines.

#### Acceptance Criteria

1. THE pgm-toolkit directory SHALL be moved to external/pgm-toolkit location
2. THE pyproject.toml SHALL include pgm-toolkit as an optional dependency or document the installation method
3. WHEN pgm-toolkit is installed via `pip install -e external/pgm-toolkit`, THE BP algorithms SHALL import successfully
4. THE repository SHALL NOT contain pgm-toolkit as a nested git repository in the root directory
5. IF pgm-toolkit is not installed, THEN THE import errors SHALL be handled gracefully with clear error messages

### Requirement 9: 根目录产物清理

**User Story:** As a developer, I want the repository root to be clean of generated artifacts, so that it follows AGENTS.md constraints.

#### Acceptance Criteria

1. THE repository root SHALL NOT contain .png files that are experiment outputs
2. WHEN experiment outputs are generated, THE outputs SHALL be saved to analysis/plots/ or analysis/stats/ directories
3. THE existing .png files in root SHALL be either deleted or moved to appropriate analysis/ subdirectories
