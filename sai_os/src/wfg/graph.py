"""Wait-for Graph (WFG) construction for deadlock modeling."""

import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Optional


class WaitForGraph:
    """
    Represents the Wait-for Graph for deadlock detection.
    
    Nodes = processes (or transactions)
    Edge P_i -> P_j means P_i is waiting for a resource held by P_j
    """

    def __init__(self):
        self._graph = nx.DiGraph()

    def add_process(self, process_id: int, priority: int = 0) -> None:
        """Add a process node to the graph."""
        self._graph.add_node(process_id, priority=priority)

    def add_wait_edge(self, waiter: int, holder: int, resource_id: Optional[int] = None) -> None:
        """
        Add edge: waiter -> holder (waiter is waiting for resource held by holder).
        """
        self._graph.add_edge(waiter, holder, resource=resource_id)

    def remove_edge(self, waiter: int, holder: int) -> None:
        """Remove a wait-for edge."""
        if self._graph.has_edge(waiter, holder):
            self._graph.remove_edge(waiter, holder)

    def get_adjacency_matrix(self) -> np.ndarray:
        """Return adjacency matrix of the WFG."""
        nodes = sorted(self._graph.nodes())
        n = len(nodes)
        adj = np.zeros((n, n))
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        for u, v in self._graph.edges():
            if u in node_to_idx and v in node_to_idx:
                adj[node_to_idx[u], node_to_idx[v]] = 1
        return adj

    def get_node_features(self) -> np.ndarray:
        """
        Extract node features: [in_degree, out_degree, priority_normalized].
        """
        nodes = sorted(self._graph.nodes())
        features = []
        priorities = [self._graph.nodes[n].get("priority", 0) for n in nodes]
        max_p = max(priorities) if priorities else 1
        for node in nodes:
            in_deg = self._graph.in_degree(node)
            out_deg = self._graph.out_degree(node)
            priority = self._graph.nodes[node].get("priority", 0) / max_p if max_p > 0 else 0
            features.append([in_deg, out_deg, priority])
        return np.array(features, dtype=np.float32)

    def get_edge_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return edge_index in COO format for PyTorch Geometric.
        Returns (2, num_edges) arrays.
        """
        nodes = sorted(self._graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        rows, cols = [], []
        for u, v in self._graph.edges():
            if u in node_to_idx and v in node_to_idx:
                rows.append(node_to_idx[u])
                cols.append(node_to_idx[v])
        return np.array([rows, cols], dtype=np.int64)

    def get_networkx_graph(self) -> nx.DiGraph:
        """Return the underlying NetworkX graph."""
        return self._graph

    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    def clear(self) -> None:
        """Clear the graph."""
        self._graph.clear()
