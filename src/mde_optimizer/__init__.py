"""Multi-population based Differential Evolution (MDE) - Karkinli (2023)."""
from .core import MDEResult, minimize

__version__ = "2.0.0"
__all__ = ["MDEResult", "minimize", "__version__"]
