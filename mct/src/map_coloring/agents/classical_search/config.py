"""
模块 1：配置层（CSPConfig）
把目前写死在代码里的策略变成显式参数
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import random


# 变量排序策略类型
VarOrderingStrategy = Literal["static", "mrv", "degree", "mrv+degree", "random"]

# 值排序策略类型
ValueOrderingStrategy = Literal["static", "random", "least_constraining"]


@dataclass
class CSPConfig:
    """CSP 求解器配置结构"""
    
    # ========== 变量排序策略 ==========
    var_ordering_strategy: VarOrderingStrategy = "static"
    """
    变量排序策略：
    - "static": 按原始顺序选择变量
    - "mrv": 最小剩余值（Minimum Remaining Values）
    - "degree": 度启发式（选择约束最多的变量）
    - "mrv+degree": MRV + 度启发式组合
    - "random": 随机选择未赋值变量
    """
    
    # ========== 值排序策略 ==========
    value_ordering_strategy: ValueOrderingStrategy = "static"
    """
    值排序策略：
    - "static": 按原始顺序尝试颜色
    - "random": 随机打乱颜色顺序
    - "least_constraining": 最少约束值（选择对邻居限制最少的值）
    """
    
    # ========== 约束传播策略 ==========
    use_forward_checking: bool = False
    """是否启用前向检查（Forward Checking）"""
    
    use_arc_consistency: bool = False
    """是否启用弧一致性（Arc Consistency）- 占位，暂未实现"""
    
    # ========== 资源限制 ==========
    max_steps: Optional[int] = None
    """最大搜索步数限制，None 表示无限制"""
    
    max_backtracks: Optional[int] = None
    """最大回溯次数限制，None 表示无限制"""
    
    find_all_solutions: bool = False
    """是否查找所有解（True）还是找到第一个解就停止（False）"""
    
    # ========== 随机性 ==========
    rng_seed: Optional[int] = None
    """随机数种子，None 表示使用系统随机"""
    
    random_tie_breaking: bool = False
    """在 MRV 等启发式下是否随机打平"""
    
    # ========== 日志选项 ==========
    log_assignment_snapshots: bool = True
    """是否记录每步的赋值快照（大图可关闭以节省内存）"""
    
    def __post_init__(self):
        """初始化后处理"""
        self._rng: Optional[random.Random] = None
        self._init_rng()
    
    def _init_rng(self):
        """初始化随机数生成器"""
        if self.rng_seed is not None:
            self._rng = random.Random(self.rng_seed)
        else:
            self._rng = random.Random()
    
    @property
    def rng(self) -> random.Random:
        """获取随机数生成器"""
        if self._rng is None:
            self._init_rng()
        return self._rng
    
    def reset_rng(self):
        """重置随机数生成器（用于可重复实验）"""
        self._init_rng()
    
    def validate(self) -> list[str]:
        """验证配置有效性，返回错误列表"""
        errors = []
        
        valid_var_strategies = ["static", "mrv", "degree", "mrv+degree", "random"]
        if self.var_ordering_strategy not in valid_var_strategies:
            errors.append(f"无效的变量排序策略: {self.var_ordering_strategy}")
        
        valid_value_strategies = ["static", "random", "least_constraining"]
        if self.value_ordering_strategy not in valid_value_strategies:
            errors.append(f"无效的值排序策略: {self.value_ordering_strategy}")
        
        if self.max_steps is not None and self.max_steps <= 0:
            errors.append(f"max_steps 必须为正整数: {self.max_steps}")
        
        if self.max_backtracks is not None and self.max_backtracks <= 0:
            errors.append(f"max_backtracks 必须为正整数: {self.max_backtracks}")
        
        return errors
    
    def copy(self) -> 'CSPConfig':
        """创建配置副本"""
        return CSPConfig(
            var_ordering_strategy=self.var_ordering_strategy,
            value_ordering_strategy=self.value_ordering_strategy,
            use_forward_checking=self.use_forward_checking,
            use_arc_consistency=self.use_arc_consistency,
            max_steps=self.max_steps,
            max_backtracks=self.max_backtracks,
            find_all_solutions=self.find_all_solutions,
            rng_seed=self.rng_seed,
            random_tie_breaking=self.random_tie_breaking,
            log_assignment_snapshots=self.log_assignment_snapshots,
        )


# 预定义配置模板
DEFAULT_CONFIG = CSPConfig()

FAST_CONFIG = CSPConfig(
    var_ordering_strategy="mrv",
    value_ordering_strategy="least_constraining",
    use_forward_checking=True,
    log_assignment_snapshots=False,
)

RANDOM_CONFIG = CSPConfig(
    var_ordering_strategy="random",
    value_ordering_strategy="random",
    rng_seed=42,
)
