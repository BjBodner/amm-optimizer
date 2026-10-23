"""Adaptive Multi Mode (AMM).

Budget-aware routing of optimizers by online landscape probes: a derivative-free
black-box optimizer.
"""

from . import presets
from .api import minimize
from .config import ATMConfig
from .profiling import Profiler, TimingProfiler, Timings
from .result import OptimizeResult

__version__ = "0.3.0"
__all__ = [
    "ATMConfig",
    "OptimizeResult",
    "Profiler",
    "Timings",
    "TimingProfiler",
    "minimize",
    "presets",
    "__version__",
]
