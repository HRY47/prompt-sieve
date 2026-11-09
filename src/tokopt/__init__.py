"""tokopt - token-aware prompt optimizer."""
from .counters import get_counter, TokenCounter, HeuristicCounter
from .optimizer import optimize, OptimizeResult, Candidate
from .check import preservation, chrf

__all__ = ["get_counter", "TokenCounter", "HeuristicCounter", "optimize",
           "OptimizeResult", "Candidate", "preservation", "chrf"]
__version__ = "0.1.0"

from .runtime import OptimizationCache
__all__.append("OptimizationCache")
