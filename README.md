# Hybrid WFG-GNN Deadlock Avoidance System

An intelligent deadlock avoidance framework that combines traditional **Wait-For Graph (WFG)** cycle detection using the **Chandy-Misra algorithm** with a **Graph Neural Network (GNN)** for predictive deadlock risk analysis.

This project demonstrates how classical operating system concepts can be integrated with modern AI techniques to build scalable and intelligent deadlock avoidance systems for distributed environments.

---

# Features

* Traditional Wait-For Graph based deadlock detection
* Chandy-Misra distributed cycle detection algorithm
* Graph Neural Network for deadlock risk prediction
* Hybrid decision-making engine
* Synthetic graph generation for training/testing
* Support for CPU and GPU execution
* Modular architecture for research and experimentation
* Easy training and evaluation pipeline

---

# System Architecture

## Workflow

1. Build a Wait-For Graph (WFG)
2. Detect cycles using Chandy-Misra algorithm
3. Convert graph into GNN-compatible format
4. Predict deadlock probability using Graph Neural Network
5. Hybrid engine makes allocation decision

---

# Project Structure

```text
s_os/
│
├── checkpoints/
│   └── gnn_model.pt              # Saved trained model
│
├── src/
│   ├── datasets/
│   │   ├── graph_loader.py       # Graph conversion utilities
│   │   ├── synthetic.py          # Synthetic WFG generator
│   │   └── __init__.py
│   │
│   ├── gnn/
│   │   ├── model.py              # GNN model architecture
│   │   └── __init__.py
│   │
│   ├── hybrid/
│   │   ├── decision.py           # Hybrid decision engine
│   │   └── __init__.py
│   │
│   ├── wfg/
│   │   ├── chandy_misra.py       # Chandy-Misra algorithm
│   │   ├── graph.py              # Wait-For Graph implementation
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── main.py                       # Main demonstration script
├── train.py                      # Model training script
├── requirements.txt              # Required dependencies
└── README.md
```

---

# Technologies Used

| Technology        | Purpose                   |
| ----------------- | ------------------------- |
| Python            | Core programming language |
| PyTorch           | Deep learning framework   |
| PyTorch Geometric | Graph Neural Networks     |
| NetworkX          | Graph operations          |
| NumPy             | Numerical computing       |
| Scikit-learn      | ML utilities              |
| Matplotlib        | Visualization             |
| tqdm              | Progress tracking         |

---

# Installation Guide

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/s_os.git
cd s_os
```

---

## 2. Create Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dependencies

The project requires the following Python packages:

```text
torch>=2.0.0
torch-geometric>=2.3.0
networkx>=3.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

---

# PyTorch Geometric Installation

Sometimes `torch-geometric` may require additional setup.

## Install PyTorch

### CPU Version

```bash
pip install torch torchvision torchaudio
```

### CUDA Version

Visit:

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

and install the version matching your CUDA.

---

## Install PyTorch Geometric

```bash
pip install torch-geometric
```

If installation fails, use:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric
```

---

# How to Run

## Run the Complete Demo

```bash
python main.py
```

This demonstrates:

* Traditional WFG deadlock detection
* Hybrid WFG + GNN prediction
* Manual cycle creation and detection

---

# Train the GNN Model

```bash
python train.py --epochs 50 --samples 2000 --max-nodes 100 --save checkpoints/gnn_model.pt
```

---

# Training Parameters

| Argument      | Description                       |
| ------------- | --------------------------------- |
| `--epochs`    | Number of training epochs         |
| `--samples`   | Number of synthetic graph samples |
| `--max-nodes` | Maximum nodes per graph           |
| `--save`      | Path to save trained model        |

---

# Example Usage

```python
from src.wfg import WaitForGraph, ChandyMisraDetector
from src.gnn import DeadlockGNN
from src.hybrid import HybridDeadlockAvoidance

# Create wait-for graph
wfg = WaitForGraph()

wfg.add_process(0, priority=5)
wfg.add_process(1, priority=3)

# Process 0 waits for Process 1
wfg.add_wait_edge(0, 1)

# Traditional detection
cm = ChandyMisraDetector(wfg)
has_cycle, cycles = cm.detect()

print(has_cycle)
print(cycles)

# GNN model
model = DeadlockGNN(
    in_channels=3,
    hidden_channels=64,
    num_layers=3
)

# Hybrid system
hybrid = HybridDeadlockAvoidance(
    gnn_model=model,
    risk_threshold=0.5
)

allow, reason = hybrid.should_allow_allocation(wfg)

print(allow)
print(reason)
```

---

# Expected Output

## Traditional WFG

```text
Has cycle (potential deadlock): True
Cycles found: 1
Example cycle: [0, 1, 2]
```

---

## Hybrid WFG + GNN

```text
Case 1 (forced deadlock):
Cycle: True
Risk: 0.812
Allow allocation: False
```

---

## Manual WFG

```text
Manual cycle 0->1->2->0:
has_deadlock=True
```

---

# Hybrid Decision Logic

The allocation is denied only when:

* A cycle exists in the Wait-For Graph
  AND
* Predicted deadlock risk exceeds threshold

## Decision Formula

```text
IF cycle_detected == True
AND risk_score > threshold
THEN deny allocation
ELSE allow allocation
```

---

# Research Applications

This project can be extended for:

* Distributed Operating Systems
* Cloud Resource Management
* AI-assisted Scheduling
* Smart Resource Allocation
* Parallel Computing
* High Performance Computing (HPC)
* Deadlock Prediction Research
* Intelligent OS Simulation

---

# Future Improvements

* Real-time distributed simulation
* Reinforcement learning based allocation
* Dynamic graph streaming
* Kubernetes resource deadlock prediction
* Multi-agent deadlock resolution
* Explainable AI for deadlock prediction
* Web dashboard visualization

---

# Troubleshooting

## ModuleNotFoundError

Install dependencies again:

```bash
pip install -r requirements.txt
```

---

## Torch Geometric Installation Issues

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Then reinstall PyTorch and torch-geometric.

---

## CUDA Not Detected

Check GPU availability:

```python
import torch
print(torch.cuda.is_available())
```

---

# Academic Relevance

This project is suitable for:

* M.E / M.Tech Operating Systems projects
* AI + Systems research
* GNN research demonstrations
* Distributed systems simulation
* Final year engineering projects

---

# Author

Sai Deshiga Sri S

---

# License

This project is open-source and available for academic and research purposes.

---

# Contributing

Pull requests and improvements are welcome.

Steps:

1. Fork the repository
2. Create a new branch
3. Commit changes
4. Push branch
5. Open Pull Request

---

# Support

If you found this project useful:

* Star the repository
* Share with researchers and students
* Use it for learning and experimentation

---

# Keywords

Deadlock Avoidance, Wait-For Graph, Chandy-Misra Algorithm, Graph Neural Network, Distributed Systems, Operating Systems, AI for Systems, GNN, PyTorch Geometric, Hybrid Deadlock Detection
