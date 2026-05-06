"""DataLoader for converting WFGs to PyTorch Geometric batch."""

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from typing import List, Tuple
from .synthetic import SyntheticGraphDataset
from ..wfg import WaitForGraph


class GraphDataset(Dataset):
    """PyTorch Dataset wrapping WFG graphs and labels."""

    def __init__(self, wfg_list: List[WaitForGraph], labels: List[int]):
        self.wfg_list = wfg_list
        self.labels = labels

    def __len__(self) -> int:
        return len(self.wfg_list)

    def __getitem__(self, idx: int) -> Tuple[Data, int]:
        wfg = self.wfg_list[idx]
        x = torch.tensor(wfg.get_node_features(), dtype=torch.float32)
        edge_index = torch.tensor(wfg.get_edge_index(), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        label = self.labels[idx]
        return data, label


def collate_fn(batch: List[Tuple[Data, int]]) -> Tuple[Data, torch.Tensor]:
    """Collate to batch Data and labels."""
    data_list = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.float32).unsqueeze(1)
    batched = Batch.from_data_list(data_list)
    return batched, labels


class GraphDataLoader:
    """Create DataLoaders from SyntheticGraphDataset."""

    @staticmethod
    def from_synthetic(
        dataset: SyntheticGraphDataset,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        shuffle: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        graphs = dataset.graphs
        labels = dataset.labels
        n = len(graphs)
        split = int(n * train_ratio)
        train_ds = GraphDataset(graphs[:split], labels[:split])
        val_ds = GraphDataset(graphs[split:], labels[split:])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return train_loader, val_loader
