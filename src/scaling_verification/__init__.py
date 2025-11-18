"""
Alias package so ``python -m scaling_verification.selection.run`` works
without installing the upstream dependency.
"""

from . import selection

__all__ = ["selection"]
