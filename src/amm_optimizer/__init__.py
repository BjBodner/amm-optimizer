"""Adaptive Multi Mode (AMM).

Budget-aware routing of optimizers by online landscape probes: a derivative-free
black-box optimizer.
"""

# ``api.minimize`` is the engine; its default is the original ATM sampler.
from . import api as api
from . import presets
from .config import ATMConfig
from .default import minimize
from .profiling import Profiler, TimingProfiler, Timings
from .result import OptimizeResult

__version__ = "0.7.0"
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
