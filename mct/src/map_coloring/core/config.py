"""Configuration management using YAML + dataclass."""

from dataclasses import dataclass, field
from typing import Optional
import os
from datetime import datetime
from pathlib import Path


@dataclass
class CSPConfigClass:
    """CSP solver configuration."""
    variable_ordering: str = "mrv"
    value_ordering: str = "least_constraining"
    forward_checking: bool = True
    arc_consistency: bool = False
    max_backtracks: int = 1000
    rng_seed: int = 42


@dataclass
class BPConfigClass:
    """Belief Propagation configuration."""
    max_iterations: int = 100
    tolerance: float = 1.0e-3
    threshold_margin: float = 0.1
    confidence_gap: float = 0.0
    damping: float = 0.0
    same_color_penalty: float = 1.0e-6
    diff_color_reward: float = 1.0
    update_schedule: str = "synchronous"  # "synchronous" 或 "asynchronous"


@dataclass
class OutputConfig:
    """Output configuration."""
    stats_dir: str = "analysis/stats"
    plots_dir: str = "analysis/plots"
    save_plots: bool = True
    save_metrics: bool = True


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    csp: CSPConfigClass = field(default_factory=CSPConfigClass)
    bp: BPConfigClass = field(default_factory=BPConfigClass)
    output: OutputConfig = field(default_factory=OutputConfig)

    @staticmethod
    def generate_run_id() -> str:
        """Generate run ID in format YYYYMMDD-HHMMSS."""
        return datetime.now().strftime("%Y%m%d-%H%M%S")


def _find_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume src/map_coloring/core/config.py structure
    return current.parent.parent.parent.parent


def load_config(config_path: Optional[str] = None) -> ExperimentConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses default config.

    Returns:
        ExperimentConfig instance
    """
    config = ExperimentConfig()

    if config_path is None:
        # Use default config path
        project_root = _find_project_root()
        config_path = str(project_root / "configs" / "default.yaml")

    if not os.path.exists(config_path):
        return config

    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        # Update CSP config
        if 'csp' in data:
            for key, value in data['csp'].items():
                if hasattr(config.csp, key):
                    setattr(config.csp, key, value)

        # Update BP config
        if 'bp' in data:
            for key, value in data['bp'].items():
                if hasattr(config.bp, key):
                    setattr(config.bp, key, value)

        # Update Output config
        if 'output' in data:
            for key, value in data['output'].items():
                if hasattr(config.output, key):
                    setattr(config.output, key, value)

    except ImportError:
        # PyYAML not installed, use defaults
        pass
    except Exception:
        # Error loading YAML, use defaults
        pass

    return config


def get_output_path(run_id: str, task_name: str = "experiment1_map_coloring",
                    config: Optional[ExperimentConfig] = None) -> dict[str, str]:
    """
    Get output paths for a run.

    Args:
        run_id: Run identifier
        task_name: Task name
        config: Experiment configuration

    Returns:
        Dictionary with 'stats_dir' and 'plots_dir' paths
    """
    if config is None:
        config = load_config()

    stats_dir = os.path.join(config.output.stats_dir, task_name, run_id)
    plots_dir = os.path.join(config.output.plots_dir, task_name, run_id)

    return {
        "stats_dir": stats_dir,
        "plots_dir": plots_dir,
    }
