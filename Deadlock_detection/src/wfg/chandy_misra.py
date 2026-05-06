"""Chandy-Misra algorithm for deadlock detection in distributed systems."""

import networkx as nx
from typing import List, Optional, Tuple
from .graph import WaitForGraph


class ChandyMisraDetector:
    """
    Chandy-Misra algorithm for detecting cycles (potential deadlocks) in WFG.
    
    Cycle in WFG => potential deadlock.
    """

    def __init__(self, wfg: WaitForGraph):
        self._wfg = wfg

    def has_cycle(self) -> bool:
        """Return True if the WFG contains a cycle (potential deadlock)."""
        G = self._wfg.get_networkx_graph()
        try:
            cycle = nx.find_cycle(G)
            return len(cycle) > 0
        except nx.NetworkXNoCycle:
            return False

    def get_cycles(self) -> List[List[int]]:
        """
        Return list of cycles. Each cycle is a list of node IDs.
        Uses DFS-based cycle enumeration.
        """
        G = self._wfg.get_networkx_graph()
        cycles = []
        try:
            cycle_edges = list(nx.simple_cycles(G))
            for edge_list in cycle_edges:
                cycles.append(edge_list)
        except Exception:
            pass
        return cycles

    def detect(self) -> Tuple[bool, List[List[int]]]:
        """
        Run detection. Returns (has_deadlock, list_of_cycles).
        """
        cycles = self.get_cycles()
        return len(cycles) > 0, cycles
