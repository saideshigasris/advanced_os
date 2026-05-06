"""Hybrid WFG-GNN decision logic for deadlock avoidance."""

import numpy as np
import torch
from typing import Tuple, Optional

from ..wfg import WaitForGraph, ChandyMisraDetector
from ..gnn import DeadlockGNN
from torch_geometric.data import Data


class HybridDeadlockAvoidance:
    """
    Hybrid framework: WFG cycle detection + GNN risk scoring.
    
    - WFG (Chandy-Misra) detects cycles
    - GNN predicts deadlock risk
    - Decision: high risk -> deny allocation; low risk -> allow
    """

    def __init__(
        self,
        gnn_model: Optional[DeadlockGNN] = None,
        risk_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.gnn = gnn_model
        self.risk_threshold = risk_threshold
        self.device = device
        if self.gnn is not None:
            self.gnn = self.gnn.to(device)
            self.gnn.eval()

    def wfg_to_tensor_data(self, wfg: WaitForGraph) -> Data:
        """Convert WFG to PyTorch Geometric Data."""
        x = torch.tensor(wfg.get_node_features(), dtype=torch.float32)
        edge_index = torch.tensor(wfg.get_edge_index(), dtype=torch.long)
        return Data(x=x, edge_index=edge_index, batch=torch.zeros(x.size(0), dtype=torch.long))

    def evaluate(self, wfg: WaitForGraph) -> Tuple[bool, float, bool]:
        """
        Evaluate a WFG and return decision.
        
        Returns:
            (has_cycle, risk_score, allow_allocation)
            - allow_allocation: True if we allow the allocation, False to deny
        """
        detector = ChandyMisraDetector(wfg)
        has_cycle, cycles = detector.detect()

        if not has_cycle:
            return False, 0.0, True  # No cycle -> no deadlock risk -> allow

        risk_score = 0.0
        if self.gnn is not None:
            data = self.wfg_to_tensor_data(wfg).to(self.device)
            with torch.no_grad():
                risk_score = self.gnn(data).item()

        # Hybrid decision: deny if cycle AND risk above threshold
        allow = not (has_cycle and risk_score >= self.risk_threshold)
        return has_cycle, risk_score, allow

    def should_allow_allocation(self, wfg: WaitForGraph) -> Tuple[bool, str]:
        """
        High-level API: should we allow this allocation?
        Returns (allow, reason_string).
        """
        has_cycle, risk, allow = self.evaluate(wfg)
        if not has_cycle:
            return True, "No cycle detected"
        if allow:
            return True, f"Cycle detected but low risk ({risk:.3f} < {self.risk_threshold})"
        return False, f"Cycle detected with high risk ({risk:.3f} >= {self.risk_threshold})"
