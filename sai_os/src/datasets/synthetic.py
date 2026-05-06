"""Synthetic wait-for graph generation for training and evaluation."""

import random
import numpy as np
from typing import List, Tuple, Optional
from ..wfg import WaitForGraph


def generate_synthetic_wfg(
    num_processes: int = 20,
    num_resources: int = 10,
    edge_prob: float = 0.15,
    seed: Optional[int] = None,
    force_deadlock: Optional[bool] = None,
) -> Tuple[WaitForGraph, bool]:
    """
    Generate a synthetic wait-for graph.
    
    Args:
        num_processes: Number of process nodes
        num_resources: Number of shared resources
        edge_prob: Probability of adding wait-for edge
        seed: Random seed for reproducibility
        force_deadlock: If True, ensure cycle exists; if False, try to avoid
    
    Returns:
        (WaitForGraph, has_deadlock_label)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    wfg = WaitForGraph()
    for p in range(num_processes):
        wfg.add_process(p, priority=random.randint(0, 10))

    # Build resource allocation: resource_id -> holder_process
    resource_holders = {r: random.randint(0, num_processes - 1) for r in range(num_resources)}

    if force_deadlock is True:
        # Force a cycle: 0->1->2->...->k->0
        k = min(5, num_processes - 1)
        for i in range(k):
            wfg.add_wait_edge(i, (i + 1) % (k + 1), resource_id=i % num_resources)
        # Add more random edges
        for _ in range(num_processes):
            u, v = random.sample(range(num_processes), 2)
            if u != v:
                wfg.add_wait_edge(u, v, resource_id=random.randint(0, num_resources - 1))
        has_deadlock = True
    elif force_deadlock is False:
        # DAG: only add edges from lower to higher index (no back edges)
        for u in range(num_processes):
            for v in range(u + 1, num_processes):
                if random.random() < edge_prob:
                    wfg.add_wait_edge(u, v, resource_id=random.randint(0, num_resources - 1))
        has_deadlock = False
    else:
        # Random
        for _ in range(int(num_processes * num_processes * edge_prob)):
            u, v = random.sample(range(num_processes), 2)
            if u != v:
                wfg.add_wait_edge(u, v, resource_id=random.randint(0, num_resources - 1))
        from ..wfg import ChandyMisraDetector
        has_deadlock = ChandyMisraDetector(wfg).has_cycle()

    return wfg, has_deadlock


class SyntheticGraphDataset:
    """
    Dataset of synthetic wait-for graphs with deadlock labels.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        min_processes: int = 10,
        max_processes: int = 50,
        balanced: bool = True,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.min_processes = min_processes
        self.max_processes = max_processes
        self.balanced = balanced
        self.seed = seed
        self.graphs: List[WaitForGraph] = []
        self.labels: List[int] = []

        self._generate()

    def _generate(self) -> None:
        half = self.num_samples // 2
        for i in range(self.num_samples):
            n = random.randint(self.min_processes, self.max_processes)
            force = True if (self.balanced and i < half) else (False if self.balanced and i >= half else None)
            wfg, label = generate_synthetic_wfg(
                num_processes=n,
                num_resources=max(5, n // 2),
                edge_prob=0.1 + random.random() * 0.15,
                seed=self.seed + i,
                force_deadlock=force,
            )
            self.graphs.append(wfg)
            self.labels.append(1 if label else 0)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Tuple[WaitForGraph, int]:
        return self.graphs[idx], self.labels[idx]
