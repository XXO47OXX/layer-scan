"""Inference backends for layer-scan.

Each backend wraps a specific inference framework and implements
the forward_with_duplication interface.
"""

from layer_scan.backends.base import Backend

__all__ = ["Backend"]
