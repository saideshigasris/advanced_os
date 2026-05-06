"""Dataset loaders and synthetic graph generation."""

from .synthetic import SyntheticGraphDataset
from .graph_loader import GraphDataLoader

__all__ = ["SyntheticGraphDataset", "GraphDataLoader"]
