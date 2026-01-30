# Map Coloring 地图着色实验

基于约束满足问题（CSP）和贝叶斯方法的地图着色实验项目。

## 项目简介

本项目实现了两类地图着色算法：

1. **经典搜索算法（CSP）**：基于约束满足问题的回溯搜索，支持多种变量选择和值排序策略
2. **贝叶斯推理算法（BP）**：基于置信传播的概率图模型方法，支持 decimation 策略

项目采用五层架构设计：
- `core/`：核心协议层（推理适配器、追踪/结果 schema）
- `agents/`：算法实现层（CSP 求解器、BP 推理）
- `tasks/`：任务与实验编排
- `analysis/`：统计分析与可视化
- `fitting/`：参数学习与拟合（预留）

## 安装说明 (Recommended)

本项目使用 **[uv](https://docs.astral.sh/uv/)** 进行高速、可复现的环境管理，并使用 **Git Submodule** 管理 `pgm-toolkit` 依赖。

> ⚠️ **重要提示**: 克隆时必须使用 `--recurse-submodules` 参数。

### 1. 克隆项目

- **新用户**:
  ```bash
  git clone --recurse-submodules <repository-url>
  cd map_coloring/mct
  ```

- **已克隆但没有子模块的用户**:
  ```bash
  # 初始化并拉取 pgm-toolkit
  git submodule update --init --recursive
  cd mct
  ```

### 2. 同步环境

确保你已经安装了 `uv` (未安装可运行 `curl -LsSf https://astral.sh/uv/install.sh | sh` 或 `pip install uv`)。

```bash
# 一键安装所有依赖（即使断网也能通过本地 submodule 安装 pgm-toolkit）
# --extra bp  : 安装贝叶斯推理与 pgm-toolkit 依赖
# --extra dev : 安装 pytest 等开发工具
uv sync --extra bp --extra dev
```

### 3. 激活环境

- **Windows**:
  ```powershell
  .venv\Scripts\activate
  ```
- **Linux/Mac**:
  ```bash
  source .venv/bin/activate
  ```

> 提示: 你也可以不激活环境，直接使用 `uv run python scripts/xxx.py` 来运行脚本。

## 快速开始

### 运行 CSP 求解器

```bash
# 快速演示
python scripts/quick_csp_demo.py

# 或在 Python 中使用
python -c "
from agents.classical_search.csp_solver import CSPSolver, CSPConfig
from tasks.experiment1_map_coloring.maps import MAPS

# 获取地图配置
map_config = MAPS['australia']

# 创建求解器
config = CSPConfig(
    var_strategy='mrv',
    val_strategy='lcv'
)
solver = CSPSolver(
    regions=map_config['regions'],
    neighbors=map_config['neighbors'],
    colors=map_config['colors'],
    config=config
)

# 求解
solution = solver.solve()
print(f'解: {solution}')
"
```

### 运行 BP 推理（需要 pgm-toolkit）

```bash
# 快速演示（如果未激活环境，请在前面加上 uv run）
python scripts/quick_bp_demo.py
```

### 运行算法对比实验

```bash
# 对比 CSP 和 BP 算法
python scripts/run_comparison.py
```

## 目录结构

```
map_coloring/
├── README.md                    # 项目说明文档
├── pyproject.toml               # 包配置
├── .gitignore                   # Git 忽略规则
├── AGENTS.md                    # 协作指令文档
│
├── core/                        # 核心协议层
│   ├── __init__.py
│   ├── config.py                # 实验配置
│   └── inference/               # 推理协议
│       ├── schemas.py           # TraceSchema, ResultSchema
│       └── adapters.py          # InferenceAdapter ABC
│
├── agents/                      # 算法实现层
│   ├── classical_search/        # CSP 搜索算法
│   │   └── csp_solver/          # CSP 求解器
│   └── bayesian/                # 贝叶斯推理
│       └── research_bp/         # BP 算法研究
│
├── tasks/                       # 任务与实验
│   └── experiment1_map_coloring/
│       ├── maps.py              # 地图数据定义
│       └── demo_csp_solver.py   # CSP 演示
│
├── analysis/                    # 分析与可视化
│   ├── stats/                   # 统计结果输出
│   └── plots/                   # 图表输出
│
├── fitting/                     # 参数学习（预留）
│
├── utils/                       # 工具函数
│
├── scripts/                     # 运行脚本
│   ├── run_comparison.py        # 算法对比
│   ├── quick_csp_demo.py        # CSP 快速演示
│   └── quick_bp_demo.py         # BP 快速演示
│
├── tests/                       # 测试套件
│
└── configs/                     # 配置文件
    └── default.yaml             # 默认配置
```

## 实验输出

实验运行产物按照 `run_id` 格式（`YYYYMMDD-HHMMSS_<note>`）组织：

```
analysis/
├── stats/<task>/<run_id>/
│   ├── metrics.json             # 关键指标
│   └── config.yaml              # 本次参数
└── plots/<task>/<run_id>/
    └── *.png                    # 可视化图表
```

## 依赖说明

- Python >= 3.10, < 3.12
项目中包含 `uv.lock` 文件，锁定了所有依赖的精确版本。

- Python >= 3.10, < 3.12
- numpy, matplotlib, networkx, pyyaml
- **pgm-toolkit**: 作为 Git submodule 本地集成

## 测试

```bash
# 运行所有测试
uv run pytest tests/

## 许可证

MIT License
