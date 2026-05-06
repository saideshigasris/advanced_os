# Hybrid WFG-GNN Deadlock Avoidance

Combines **Wait-for Graph (Chandy-Misra)** cycle detection with **Graph Neural Network** risk prediction for scalable deadlock avoidance in distributed systems.

## Architecture

1. **WFG** – Build wait-for graph; detect cycles via Chandy-Misra  
2. **GNN** – Predict deadlock risk (0–1) from graph structure  
3. **Hybrid decision** – Deny allocation only when cycle **and** high risk

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Train the GNN

```bash
python train.py --epochs 50 --samples 2000 --max-nodes 100 --save checkpoints/gnn_model.pt
```

### Run demo

```bash
python main.py
```
What “correct” means
Part	Expected behavior
Demo 1 (Traditional WFG)	Cycle is always detected for force_deadlock=True. Cycle IDs may vary.
Demo 2 Case 1 (deadlock)	Trained: high risk (e.g. 0.7+), deny allocation. Untrained: risk around 0.5, decision can be allow or deny.
Demo 2 Case 2 (no deadlock)	Cycle: False, Risk: 0.000, Allow: True always.
Demo 3 (manual cycle)	has_deadlock=True, cycles=[[0, 1, 2]] always.


### Use in code

```python
from src.wfg import WaitForGraph, ChandyMisraDetector
from src.gnn import DeadlockGNN
from src.hybrid import HybridDeadlockAvoidance

# Build WFG
wfg = WaitForGraph()
wfg.add_process(0, priority=5)
wfg.add_process(1, priority=3)
wfg.add_wait_edge(0, 1)  # 0 waits for 1

# Hybrid evaluation
gnn = DeadlockGNN(in_channels=3, hidden_channels=64)
hybrid = HybridDeadlockAvoidance(gnn_model=gnn, risk_threshold=0.5)
allow, reason = hybrid.should_allow_allocation(wfg)
```

## Project structure

```
sai_os/
├── src/
│   ├── wfg/           # Wait-for graph, Chandy-Misra
│   ├── gnn/           # Deadlock GNN model
│   ├── hybrid/        # Hybrid decision logic
│   └── datasets/      # Synthetic graph generation
├── train.py           # GNN training
├── main.py            # Demo script
└── requirements.txt
```
