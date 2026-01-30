# Design Document: map_coloring 仓库重构清理

## Overview

本设计文档描述了修复 map_coloring 仓库重构遗留问题的技术方案。目标是使仓库完全符合 AGENTS.md 定义的五层架构规范，实现"可安装、可运行、可测试"的工程标准。

主要工作包括：
1. 创建 README.md 使包可安装
2. 消除所有 sys.path 注入，改用标准包导入
3. 实现 core/inference 的 adapter 和 schema 协议
4. 建立基础测试套件
5. 修正 .gitignore 规则
6. 消除数据重复
7. 规范化 pgm-toolkit 依赖位置
8. 清理根目录产物

## Architecture

```
map_coloring/
├── README.md                    # [新增] 项目说明文档
├── pyproject.toml               # 包配置（已存在，需微调）
├── .gitignore                   # [修改] 修正忽略规则
├── external/                    # [新增] 外部依赖目录
│   └── pgm-toolkit/             # [移动] 从根目录移入
├── core/
│   ├── __init__.py
│   ├── config.py
│   └── inference/
│       ├── __init__.py
│       ├── schemas.py           # [新增] TraceSchema, ResultSchema
│       └── adapters.py          # [新增] InferenceAdapter ABC
├── agents/
│   ├── classical_search/
│   └── bayesian/
├── tasks/
│   └── experiment1_map_coloring/
│       ├── maps.py              # 唯一的地图数据源
│       └── demo_csp_solver.py   # [修改] 移除内嵌 MAPS
├── analysis/
│   ├── stats/
│   └── plots/
├── scripts/
│   ├── run_comparison.py        # [修改] 移除 sys.path 注入
│   ├── quick_csp_demo.py        # [修改] 移除 sys.path 注入
│   └── quick_bp_demo.py         # [修改] 移除 sys.path 注入
└── tests/
    ├── __init__.py
    ├── test_csp_solver.py       # [新增] CSP 求解器测试
    └── test_bp_metrics.py       # [新增] BP 指标测试
```

## Components and Interfaces

### 1. README.md

创建项目说明文档，包含：
- 项目简介
- 安装说明（包括 pgm-toolkit 可选依赖）
- 快速开始示例
- 目录结构说明

### 2. core/inference/schemas.py

```python
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime

@dataclass
class TraceStep:
    """单个推理步骤的记录"""
    step_id: int
    action: str  # 'assign', 'backtrack', 'try', 'reject', etc.
    timestamp: float
    depth: int
    var: Optional[str] = None
    value: Optional[Any] = None
    metadata: dict = field(default_factory=dict)

@dataclass
class TraceSchema:
    """推理过程追踪协议"""
    algorithm: str
    start_time: datetime
    steps: list[TraceStep] = field(default_factory=list)
    
    def add_step(self, action: str, depth: int, **kwargs) -> TraceStep:
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
    """推理结果标准化结构"""
    algorithm: str
    solved: bool
    solution: Optional[dict] = None
    elapsed_seconds: float = 0.0
    total_steps: int = 0
    metadata: dict = field(default_factory=dict)
```

### 3. core/inference/adapters.py

```python
from abc import ABC, abstractmethod
from typing import Any
from .schemas import TraceSchema, ResultSchema

class InferenceAdapter(ABC):
    """推理算法适配器抽象基类"""
    
    @abstractmethod
    def solve(self, problem: Any) -> ResultSchema:
        """执行推理并返回标准化结果"""
        pass
    
    @abstractmethod
    def get_trace(self) -> TraceSchema:
        """获取推理过程追踪"""
        pass
```

### 4. 脚本修改方案

移除 sys.path 注入后，脚本将依赖包的可安装性：

```python
# scripts/run_comparison.py (修改后)
#!/usr/bin/env python
"""Run comparison between CSP and BP algorithms."""

# 不再需要 sys.path 操作
from core import load_config, get_output_path, ExperimentConfig
from agents.classical_search.csp_solver import CSPConfig, CSPSolver
from tasks.experiment1_map_coloring.maps import MAPS, get_map
# ...
```

### 5. 测试结构

```python
# tests/test_csp_solver.py
import pytest
from agents.classical_search.csp_solver import CSPSolver, CSPConfig
from tasks.experiment1_map_coloring.maps import MAPS

def test_solution_validity():
    """验证解的有效性：相邻区域不同色"""
    ...

def test_logger_consistency():
    """验证 SearchLogger 计数一致性"""
    ...

# tests/test_bp_metrics.py
import pytest

try:
    from agents.bayesian.research_bp import compute_expected_violations
    PGM_AVAILABLE = True
except ImportError:
    PGM_AVAILABLE = False

@pytest.mark.skipif(not PGM_AVAILABLE, reason="pgm-toolkit not installed")
def test_bp_metrics():
    ...
```

### 6. .gitignore 修正

```gitignore
# Run outputs - 匹配实际的 run_id 格式 YYYYMMDD-HHMMSS
analysis/stats/**/20*/
analysis/plots/**/20*/

# 也保留原有的 run_ 前缀模式（向后兼容）
analysis/stats/*/run_*/
analysis/plots/*/run_*/
```

### 7. pgm-toolkit 迁移

```bash
# 迁移步骤
mkdir -p external
mv pgm-toolkit external/
# 更新 .gitmodules 或文档说明安装方式
```

## Data Models

### MapConfig (已存在于 maps.py)

```python
class MapConfig(TypedDict):
    regions: list[str]
    neighbors: dict[str, list[str]]
    colors: list[str]
    visual: str
```

### ExperimentConfig (已存在于 core/config.py)

保持现有结构，无需修改。

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: 脚本无 sys.path 注入

*For any* Python file in the scripts/ directory, the file content SHALL NOT contain `sys.path.insert` or `sys.path.append` statements.

**Validates: Requirements 2.1, 2.2, 2.3**

### Property 2: 脚本可执行性

*For any* script in scripts/ directory, when the package is installed via `pip install -e .`, executing the script SHALL complete without ImportError.

**Validates: Requirements 2.5**

### Property 3: CSP 解有效性（相邻不同色）

*For any* map configuration and *for any* solution produced by CSPSolver, all pairs of adjacent regions SHALL have different colors assigned.

**Validates: Requirements 4.3**

### Property 4: SearchLogger 计数一致性

*For any* CSP solving session, the SearchLogger's step_count SHALL equal the length of its steps list, and backtrack_count SHALL equal the number of 'backtrack' actions in the steps list.

**Validates: Requirements 4.4**

### Property 5: Gitignore 有效性

*For any* run_id generated in format `YYYYMMDD-HHMMSS*`, files created under `analysis/stats/<task>/<run_id>/` or `analysis/plots/<task>/<run_id>/` SHALL be ignored by git.

**Validates: Requirements 5.2**

### Property 6: 数据单一来源

*For any* module that imports MAPS, the imported MAPS object SHALL be identical (same object reference or equal content) to the MAPS defined in `tasks/experiment1_map_coloring/maps.py`.

**Validates: Requirements 6.4**

### Property 7: UTF-8 编码正确性

*For any* Chinese character in maps.py, when the file is read with UTF-8 encoding, the character SHALL be correctly decoded without replacement characters (U+FFFD) or mojibake.

**Validates: Requirements 7.2**

### Property 8: 实验输出位置正确性

*For any* experiment run that generates .png or .json output files, the files SHALL be written to paths under `analysis/plots/` or `analysis/stats/` directories, not to the repository root.

**Validates: Requirements 9.2**

## Error Handling

### 1. pgm-toolkit 缺失处理

当 pgm-toolkit 未安装时：
- BP 相关模块应在导入时捕获 ImportError
- 提供清晰的错误消息指导用户安装
- 测试应使用 `pytest.mark.skipif` 跳过相关测试

```python
# agents/bayesian/__init__.py
try:
    from pgm_toolkit import FactorGraph
    PGM_AVAILABLE = True
except ImportError:
    PGM_AVAILABLE = False
    
def require_pgm():
    if not PGM_AVAILABLE:
        raise ImportError(
            "pgm-toolkit is required for BP algorithms. "
            "Install with: pip install -e external/pgm-toolkit"
        )
```

### 2. 文件编码错误处理

读取文件时显式指定 UTF-8 编码：

```python
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
```

### 3. 配置文件缺失处理

当配置文件不存在时使用默认配置：

```python
def load_config(path: str = None) -> ExperimentConfig:
    if path is None:
        path = 'configs/default.yaml'
    if not os.path.exists(path):
        return ExperimentConfig()  # 使用默认值
    # ...
```

## Testing Strategy

### 测试框架

使用 pytest 作为测试框架，配合 hypothesis 进行属性测试。

### 单元测试

针对具体示例和边界条件：

1. **test_csp_solver.py**
   - 测试 triangle 地图（3色足够）
   - 测试 k4 地图（需要4色）
   - 测试无解情况（2色着色完全图K3）

2. **test_bp_metrics.py**
   - 测试已知图的期望违反数
   - 测试边界情况（空图、单节点）

3. **test_schemas.py**
   - 测试 TraceSchema 序列化/反序列化
   - 测试 ResultSchema 字段验证

### 属性测试

使用 hypothesis 库进行属性测试，每个测试至少运行 100 次迭代：

```python
from hypothesis import given, strategies as st, settings

@settings(max_examples=100)
@given(map_config=map_config_strategy())
def test_csp_solution_validity(map_config):
    """
    Feature: refactor-cleanup, Property 3: CSP 解有效性
    For any map configuration, solutions must have no adjacent same-color regions.
    """
    solver = CSPSolver(...)
    solution = solver.solve()
    if solution:
        for region, neighbors in map_config['neighbors'].items():
            for neighbor in neighbors:
                assert solution[region] != solution[neighbor]
```

### 测试配置

```python
# pytest.ini 或 pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow",
    "requires_pgm: marks tests requiring pgm-toolkit",
]
```

### 属性测试任务标注

每个属性测试必须包含注释引用设计文档中的属性：

- **Feature: refactor-cleanup, Property 3: CSP 解有效性**
- **Feature: refactor-cleanup, Property 4: SearchLogger 计数一致性**

