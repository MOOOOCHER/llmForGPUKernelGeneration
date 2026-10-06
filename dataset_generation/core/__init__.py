"""Core implementation for balanced CSP-DAG dataset generation."""

from .dataset import DatasetGenerator, statistics
from .generator import STAGES

__all__ = ["DatasetGenerator", "statistics", "STAGES"]
