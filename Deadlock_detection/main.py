"""
Main demo: Hybrid WFG-GNN Deadlock Avoidance.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.wfg import WaitForGraph, ChandyMisraDetector
from src.gnn import DeadlockGNN
from src.hybrid import HybridDeadlockAvoidance
from src.datasets.synthetic import generate_synthetic_wfg


def demo_without_gnn():
    """Demo using only WFG cycle detection (traditional approach)."""
    print("=" * 50)
    print("1. Traditional WFG-only (no GNN)")
    print("=" * 50)

    wfg, has_deadlock = generate_synthetic_wfg(num_processes=15, edge_prob=0.2, force_deadlock=True, seed=42)
    detector = ChandyMisraDetector(wfg)
    has_cycle, cycles = detector.detect()
    print(f"  Has cycle (potential deadlock): {has_cycle}")
    print(f"  Cycles found: {len(cycles)}")
    if cycles:
        print(f"  Example cycle: {cycles[0]}")


def demo_with_gnn(checkpoint_path: str = "checkpoints/gnn_model.pt"):
    """Demo hybrid WFG + GNN."""
    print("\n" + "=" * 50)
    print("2. Hybrid WFG-GNN (with trained GNN)")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gnn = DeadlockGNN(in_channels=3, hidden_channels=64, num_layers=3)
    if os.path.exists(checkpoint_path):
        gnn.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("  (No checkpoint found; using untrained GNN for demo)")

    hybrid = HybridDeadlockAvoidance(gnn_model=gnn, risk_threshold=0.5, device=device)

    # Case 1: Deadlock scenario
    wfg1, _ = generate_synthetic_wfg(num_processes=20, force_deadlock=True, seed=123)
    has_cycle, risk, allow = hybrid.evaluate(wfg1)
    print(f"\n  Case 1 (forced deadlock):")
    print(f"    Cycle: {has_cycle}, Risk: {risk:.3f}, Allow allocation: {allow}")
    allow_bool, reason = hybrid.should_allow_allocation(wfg1)
    print(f"    Decision: {reason}")

    # Case 2: No deadlock
    wfg2, _ = generate_synthetic_wfg(num_processes=20, force_deadlock=False, seed=456)
    has_cycle2, risk2, allow2 = hybrid.evaluate(wfg2)
    print(f"\n  Case 2 (no deadlock):")
    print(f"    Cycle: {has_cycle2}, Risk: {risk2:.3f}, Allow allocation: {allow2}")
    allow_bool2, reason2 = hybrid.should_allow_allocation(wfg2)
    print(f"    Decision: {reason2}")


def demo_manual_wfg():
    """Demo with manually constructed WFG."""
    print("\n" + "=" * 50)
    print("3. Manual WFG construction")
    print("=" * 50)

    wfg = WaitForGraph()
    wfg.add_process(0, priority=5)
    wfg.add_process(1, priority=3)
    wfg.add_process(2, priority=7)
    # Cycle: 0 -> 1 -> 2 -> 0
    wfg.add_wait_edge(0, 1)
    wfg.add_wait_edge(1, 2)
    wfg.add_wait_edge(2, 0)

    detector = ChandyMisraDetector(wfg)
    has_cycle, cycles = detector.detect()
    print(f"  Manual cycle 0->1->2->0: has_deadlock={has_cycle}, cycles={cycles}")


def main():
    print("\n*** Hybrid WFG-GNN Deadlock Avoidance Demo ***\n")
    demo_without_gnn()
    demo_with_gnn()
    demo_manual_wfg()
    print("\n" + "=" * 50)
    print("Done.")
    print("=" * 50)


if __name__ == "__main__":
    main()
